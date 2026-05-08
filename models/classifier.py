"""
models/classifier.py — Fault Category & Priority Classifier
=============================================================
Predicts the most likely fault category (BROADBAND / FIBER /
TELEPHONE / TELEVISION / OTHER) and priority (HIGH / MEDIUM / LOW)
from a fault description using TF-IDF + Logistic Regression.

SRS requirement:
  - Auto-suggest category when client reports a fault
  - Auto-suggest priority for admin assignment workflow
  - Re-trainable from labelled fault records in MySQL

Also provides:
  - Rule-based SLA breach risk assessment (complementary to ML)
  - Urgency scoring (0–100) for payment queue prioritisation

Pipeline:
  1. DataExtractor fetches historical fault text + labels
  2. FaultClassifier cleans text, vectorises with TF-IDF
  3. LogisticRegression predicts category + priority
  4. Returns prediction + confidence + top features as explanation

Usage:
    from models.classifier import FaultClassifier
    fc = FaultClassifier()
    result = fc.predict("No internet since morning, router light blinking")
    # result: {category, priority, confidence, urgency, explanation}
"""

import logging
import math
import pickle
import pathlib
import re
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger('slt_ai.classifier')

try:
    from sklearn.pipeline          import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model      import LogisticRegression
    from sklearn.multioutput       import MultiOutputClassifier
    from sklearn.model_selection   import cross_val_score
    from sklearn.preprocessing     import LabelEncoder
    _SKL_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not installed — classifier running in rule-based mode")
    _SKL_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

CAT_MODEL_PATH  = pathlib.Path(Config.MODEL_DIR) / 'classifier_category.pkl'
PRI_MODEL_PATH  = pathlib.Path(Config.MODEL_DIR) / 'classifier_priority.pkl'
META_SAVE_PATH  = pathlib.Path(Config.MODEL_DIR) / 'classifier_meta.pkl'

CATEGORIES = Config.FAULT_CATEGORIES          # ['BROADBAND','FIBER','TELEPHONE','TELEVISION','OTHER']
PRIORITIES  = ['HIGH', 'MEDIUM', 'LOW']

# ── Rule-based keyword maps (fallback + explanation) ──────────────────────────
CATEGORY_KEYWORDS = {
    'BROADBAND': [
        'internet', 'broadband', 'wifi', 'wi-fi', 'slow', 'router',
        'modem', 'speed', 'download', 'upload', 'connection', 'online',
        'blink', 'adsl', 'vdsl', 'disconnected', 'no internet',
    ],
    'FIBER': [
        'fiber', 'fibre', 'ftth', 'olt', 'ont', 'optical', 'cable cut',
        'fiber cut', 'light', 'red light', 'los', 'loss of signal',
    ],
    'TELEPHONE': [
        'telephone', 'phone', 'line', 'dial tone', 'landline', 'pstn',
        'busy', 'noise', 'static', 'crackle', 'dead line', 'no dial',
        'voice', 'call', 'ring',
    ],
    'TELEVISION': [
        'tv', 'television', 'peotv', 'peo tv', 'iptv', 'channel',
        'picture', 'signal', 'satellite', 'decoder', 'set top',
        'buffer', 'freeze', 'pixelate',
    ],
}

PRIORITY_RULES = {
    'HIGH': [
        'urgent', 'emergency', 'hospital', 'vip', 'business', 'completely down',
        'total outage', 'no service at all', 'critical', 'school', 'cannot work',
        'office', 'government', 'sla breach', 'escalat',
    ],
    'LOW': [
        'slow', 'intermittent', 'sometimes', 'occasional', 'minor', 'slight',
        'bit slow', 'small issue', 'not urgent', 'whenever possible',
    ],
}


class FaultClassifier:
    """
    Two-model pipeline:
    1. Category classifier  — TF-IDF + Logistic Regression (5-class)
    2. Priority classifier  — TF-IDF + Logistic Regression (3-class)

    Falls back to rule-based keyword matching when scikit-learn is
    unavailable or the model hasn't been trained yet.
    """

    def __init__(self):
        self._cat_pipeline: Optional[object] = None
        self._pri_pipeline: Optional[object] = None
        self._meta: dict = {}
        self._trained = False
        self._load_saved_models()

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def predict(self, description: str, context: dict = None) -> dict:
        """
        Predict fault category and priority from a text description.

        Args:
            description: Free-text fault description from the client.
            context:     Optional dict with extra signals:
                         { 'address': str, 'hour': int, 'customer_tier': str }

        Returns:
            {
              category:     str,       # e.g. 'BROADBAND'
              priority:     str,       # e.g. 'HIGH'
              catConfidence: float,    # 0.0–1.0
              priConfidence: float,    # 0.0–1.0
              urgencyScore:  int,      # 0–100 composite urgency
              explanation:   str,      # human-readable reason
              topKeywords:   [str],    # top 3 matched keywords
              slaRisk:       str,      # 'HIGH' | 'MEDIUM' | 'LOW'
            }
        """
        clean_text = self._clean_text(description or '')
        context    = context or {}

        if self._trained and _SKL_AVAILABLE:
            return self._ml_predict(clean_text, context)
        else:
            return self._rule_predict(clean_text, context)

    def predict_batch(self, descriptions: list) -> list:
        """Predict category + priority for a list of descriptions."""
        return [self.predict(d) for d in descriptions]

    def train(self, fault_df: pd.DataFrame) -> dict:
        """
        Train both classifiers from labelled fault data.

        Args:
            fault_df: DataFrame with columns [description, category, priority].

        Returns:
            Training metrics dict.
        """
        if not _SKL_AVAILABLE:
            return {'error': 'scikit-learn not installed'}

        logger.info(f"Training classifier on {len(fault_df)} labelled faults")

        df = self._prepare_training_data(fault_df)
        if df is None or len(df) < 20:
            return {'error': 'Insufficient labelled data (need ≥20 samples)'}

        texts       = df['text'].tolist()
        cat_labels  = df['category'].tolist()
        pri_labels  = df['priority'].tolist()

        # ── Build pipelines ──────────────────────────────────────────────────
        self._cat_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                min_df=2,
                sublinear_tf=True,
            )),
            ('clf', LogisticRegression(
                C=1.0,
                max_iter=500,
                class_weight='balanced',
                random_state=Config.KMEANS_RANDOM_STATE,
            )),
        ])

        self._pri_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=3000,
                min_df=2,
                sublinear_tf=True,
            )),
            ('clf', LogisticRegression(
                C=0.5,
                max_iter=500,
                class_weight='balanced',
                random_state=Config.KMEANS_RANDOM_STATE,
            )),
        ])

        # ── Fit ──────────────────────────────────────────────────────────────
        self._cat_pipeline.fit(texts, cat_labels)
        self._pri_pipeline.fit(texts, pri_labels)
        self._trained = True

        # ── Cross-validation accuracy ─────────────────────────────────────────
        metrics = self._cross_val_metrics(texts, cat_labels, pri_labels)
        self._meta = {
            'training_rows': len(df),
            'trained_at':    datetime.utcnow().isoformat() + 'Z',
            'metrics':       metrics,
            'categories':    sorted(set(cat_labels)),
            'priorities':    sorted(set(pri_labels)),
        }

        self._save_models()
        logger.info(
            f"Classifier trained — cat_acc={metrics.get('cat_accuracy')}%, "
            f"pri_acc={metrics.get('pri_accuracy')}%"
        )
        return metrics

    def train_from_db(self) -> dict:
        """
        Convenience: fetch labelled data from MySQL and train.
        Used by POST /api/ai/retrain.
        """
        from config import get_db_engine
        from sqlalchemy import text

        engine = get_db_engine()
        if engine is None:
            return self._train_from_synthetic()

        try:
            sql = text("""
                SELECT
                    CONCAT(
                        COALESCE(description, ''),
                        ' ',
                        COALESCE(address, '')
                    )                   AS description,
                    category,
                    priority
                FROM faults
                WHERE
                    category IS NOT NULL
                    AND priority IS NOT NULL
                    AND description IS NOT NULL
                    AND LENGTH(TRIM(description)) > 5
                ORDER BY created_at DESC
                LIMIT 5000
            """)
            df = pd.read_sql(sql, engine)
            if len(df) < 20:
                logger.info("Too few labelled faults in DB — using synthetic training data")
                return self._train_from_synthetic()
            return self.train(df)
        except Exception as exc:
            logger.error(f"DB training fetch failed: {exc}")
            return self._train_from_synthetic()

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE — ML PREDICTION
    # ─────────────────────────────────────────────────────────────────────────

    def _ml_predict(self, clean_text: str, context: dict) -> dict:
        """Full ML prediction using fitted pipelines."""
        try:
            # Category prediction
            cat_probs   = self._cat_pipeline.predict_proba([clean_text])[0]
            cat_classes = self._cat_pipeline.classes_
            cat_idx     = int(np.argmax(cat_probs))
            category    = cat_classes[cat_idx]
            cat_conf    = round(float(cat_probs[cat_idx]), 3)

            # Priority prediction
            pri_probs   = self._pri_pipeline.predict_proba([clean_text])[0]
            pri_classes = self._pri_pipeline.classes_
            pri_idx     = int(np.argmax(pri_probs))
            priority    = pri_classes[pri_idx]
            pri_conf    = round(float(pri_probs[pri_idx]), 3)

            # Priority override from context
            priority, pri_conf = self._apply_context_rules(
                clean_text, priority, pri_conf, context
            )

            # Urgency score (0–100 composite)
            urgency = self._urgency_score(cat_conf, pri_conf, priority, clean_text)

            # Top keywords (from TF-IDF feature names)
            top_kw = self._extract_top_keywords(clean_text, category)

            return {
                'category':      category,
                'priority':      priority,
                'catConfidence': cat_conf,
                'priConfidence': pri_conf,
                'urgencyScore':  urgency,
                'explanation':   self._explanation(category, priority, cat_conf, top_kw),
                'topKeywords':   top_kw,
                'slaRisk':       self._sla_risk(priority, urgency),
                'method':        'ml',
            }
        except Exception as exc:
            logger.warning(f"ML prediction failed, falling back to rules: {exc}")
            return self._rule_predict(clean_text, context)

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE — RULE-BASED PREDICTION
    # ─────────────────────────────────────────────────────────────────────────

    def _rule_predict(self, clean_text: str, context: dict) -> dict:
        """Keyword-based fallback classifier."""
        # Category: score each category by keyword hits
        cat_scores = {}
        matched_kw = {}
        for cat, keywords in CATEGORY_KEYWORDS.items():
            hits = [kw for kw in keywords if kw in clean_text]
            cat_scores[cat] = len(hits)
            matched_kw[cat] = hits

        if max(cat_scores.values(), default=0) == 0:
            category = 'BROADBAND'   # SLT's most common fault type
            cat_conf = 0.45
            top_kw   = []
        else:
            category = max(cat_scores, key=cat_scores.get)
            total_hits = sum(cat_scores.values())
            cat_conf = round(cat_scores[category] / max(total_hits, 1), 2)
            top_kw   = matched_kw[category][:3]

        # Priority: check HIGH/LOW keywords
        priority = 'MEDIUM'
        pri_conf = 0.60
        for kw in PRIORITY_RULES['HIGH']:
            if kw in clean_text:
                priority = 'HIGH'
                pri_conf = 0.75
                break
        if priority == 'MEDIUM':
            for kw in PRIORITY_RULES['LOW']:
                if kw in clean_text:
                    priority = 'LOW'
                    pri_conf = 0.65
                    break

        priority, pri_conf = self._apply_context_rules(clean_text, priority, pri_conf, context)
        urgency = self._urgency_score(cat_conf, pri_conf, priority, clean_text)

        return {
            'category':      category,
            'priority':      priority,
            'catConfidence': cat_conf,
            'priConfidence': pri_conf,
            'urgencyScore':  urgency,
            'explanation':   self._explanation(category, priority, cat_conf, top_kw),
            'topKeywords':   top_kw,
            'slaRisk':       self._sla_risk(priority, urgency),
            'method':        'rules',
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE — HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        """Lowercase, strip punctuation, normalise whitespace."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _apply_context_rules(
        self, text: str, priority: str, conf: float, context: dict
    ) -> tuple:
        """
        Apply domain-specific context overrides to priority.
        E.g., business hours + commercial address → raise priority.
        """
        hour = context.get('hour', datetime.now().hour)

        # Escalate if customer mentions hospital/school/government
        for term in ['hospital', 'school', 'government', 'police', 'bank']:
            if term in text:
                return 'HIGH', min(conf + 0.15, 0.95)

        # Escalate if reported during peak business hours (8am–6pm weekday)
        if priority == 'MEDIUM' and 8 <= hour <= 18:
            return 'MEDIUM', conf    # keep medium but note business hours

        return priority, conf

    def _urgency_score(
        self, cat_conf: float, pri_conf: float, priority: str, text: str
    ) -> int:
        """
        Composite urgency score 0–100 for payment queue sorting.

        Factors:
          - Priority weight:      HIGH=60, MEDIUM=30, LOW=10
          - Combined confidence:  up to 30 points
          - Keyword escalators:   up to 10 points
        """
        base = {'HIGH': 60, 'MEDIUM': 30, 'LOW': 10}.get(priority, 30)
        conf_bonus = round((cat_conf + pri_conf) / 2 * 30)
        kw_bonus = sum(
            5 for kw in ['hospital', 'business', 'office', 'urgent', 'emergency']
            if kw in text
        )
        return min(100, base + conf_bonus + kw_bonus)

    def _sla_risk(self, priority: str, urgency: int) -> str:
        """
        Estimate SLA breach risk.
        HIGH priority or urgency ≥ 70 → HIGH risk.
        """
        if priority == 'HIGH' or urgency >= 70:
            return 'HIGH'
        if priority == 'MEDIUM' or urgency >= 40:
            return 'MEDIUM'
        return 'LOW'

    def _explanation(
        self, category: str, priority: str, confidence: float, keywords: list
    ) -> str:
        """Generate a human-readable explanation for the prediction."""
        kw_str = ', '.join(f'"{k}"' for k in keywords[:3]) if keywords else 'no strong keywords'
        conf_str = f"{round(confidence * 100)}%"
        return (
            f"Classified as {category} ({conf_str} confidence) based on {kw_str}. "
            f"Priority set to {priority}."
        )

    def _extract_top_keywords(self, text: str, category: str) -> list:
        """Return top 3 keywords that matched the predicted category."""
        keywords = CATEGORY_KEYWORDS.get(category, [])
        return [kw for kw in keywords if kw in text][:3]

    def _prepare_training_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Clean and validate a training DataFrame."""
        df = df.copy()
        # Normalise column names
        df.columns = [c.lower().strip() for c in df.columns]
        if 'description' not in df.columns or 'category' not in df.columns:
            return None
        if 'priority' not in df.columns:
            df['priority'] = 'MEDIUM'

        df['text']     = df['description'].fillna('').apply(self._clean_text)
        df['category'] = df['category'].str.upper().str.strip()
        df['priority'] = df['priority'].str.upper().str.strip()

        # Filter to known labels
        df = df[df['category'].isin(CATEGORIES)]
        df = df[df['priority'].isin(PRIORITIES)]
        df = df[df['text'].str.len() > 3]
        df = df.drop_duplicates(subset=['text'])

        return df.reset_index(drop=True) if len(df) >= 20 else None

    def _cross_val_metrics(
        self, texts: list, cat_labels: list, pri_labels: list
    ) -> dict:
        """5-fold cross-validation accuracy for both classifiers."""
        try:
            n_splits = min(5, len(texts) // 4)
            if n_splits < 2:
                return {'cat_accuracy': None, 'pri_accuracy': None}

            cat_scores = cross_val_score(
                self._cat_pipeline, texts, cat_labels,
                cv=n_splits, scoring='accuracy'
            )
            pri_scores = cross_val_score(
                self._pri_pipeline, texts, pri_labels,
                cv=n_splits, scoring='accuracy'
            )
            return {
                'cat_accuracy':    round(float(cat_scores.mean()) * 100, 1),
                'cat_accuracy_std':round(float(cat_scores.std())  * 100, 1),
                'pri_accuracy':    round(float(pri_scores.mean()) * 100, 1),
                'pri_accuracy_std':round(float(pri_scores.std())  * 100, 1),
                'n_training':      len(texts),
                'n_splits':        n_splits,
            }
        except Exception as exc:
            logger.warning(f"Cross-val error: {exc}")
            return {'cat_accuracy': None, 'pri_accuracy': None}

    def _train_from_synthetic(self) -> dict:
        """Generate synthetic labelled faults for initial training."""
        logger.info("Generating synthetic training data for classifier")
        from data.synthetic_data import SyntheticDataGenerator
        synth = SyntheticDataGenerator(seed=42)

        # Generate realistic description-label pairs
        samples = []
        templates = {
            'BROADBAND': [
                ('internet not working since this morning', 'HIGH'),
                ('very slow broadband speed cannot stream', 'MEDIUM'),
                ('wifi keeps disconnecting every few minutes', 'MEDIUM'),
                ('no internet connection at all', 'HIGH'),
                ('router blinking red light broadband down', 'HIGH'),
                ('download speed very slow only 1mbps', 'LOW'),
                ('intermittent internet connection sometimes works', 'LOW'),
                ('cannot connect to internet office cannot work', 'HIGH'),
            ],
            'FIBER': [
                ('fiber cable cut no service', 'HIGH'),
                ('optical line red light ONT device', 'HIGH'),
                ('fiber connection completely down', 'HIGH'),
                ('LOS alarm on fiber equipment', 'HIGH'),
                ('FTTH service disrupted light blinking', 'MEDIUM'),
            ],
            'TELEPHONE': [
                ('no dial tone on landline phone', 'MEDIUM'),
                ('telephone line has noise static crackle', 'LOW'),
                ('cannot make calls dead line', 'MEDIUM'),
                ('telephone busy all the time', 'LOW'),
                ('line cuts off during calls', 'MEDIUM'),
            ],
            'TELEVISION': [
                ('peotv channels not working no picture', 'MEDIUM'),
                ('tv signal lost all channels gone', 'HIGH'),
                ('set top box not responding remote not working', 'LOW'),
                ('picture freezing and pixelating on tv', 'LOW'),
                ('peotv iptv service down', 'MEDIUM'),
            ],
            'OTHER': [
                ('billing issue wrong charge on account', 'LOW'),
                ('account suspended need to reconnect', 'MEDIUM'),
                ('want to upgrade my connection package', 'LOW'),
                ('new connection application status', 'LOW'),
            ],
        }

        rows = []
        for category, items in templates.items():
            for desc, priority in items:
                # Add variations
                for _ in range(8):
                    rows.append({
                        'description': desc,
                        'category':    category,
                        'priority':    priority,
                    })

        df = pd.DataFrame(rows)
        return self.train(df)

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE — PERSISTENCE
    # ─────────────────────────────────────────────────────────────────────────

    def _save_models(self) -> None:
        try:
            CAT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CAT_MODEL_PATH, 'wb') as f: pickle.dump(self._cat_pipeline, f)
            with open(PRI_MODEL_PATH, 'wb') as f: pickle.dump(self._pri_pipeline, f)
            with open(META_SAVE_PATH, 'wb') as f: pickle.dump(self._meta, f)
            logger.info("Classifier models saved")
        except Exception as exc:
            logger.warning(f"Classifier save failed: {exc}")

    def _load_saved_models(self) -> None:
        if not all(p.exists() for p in [CAT_MODEL_PATH, PRI_MODEL_PATH, META_SAVE_PATH]):
            logger.info("No saved classifier — will train on first request")
            # Auto-train from synthetic data on startup
            try:
                self._train_from_synthetic()
            except Exception as exc:
                logger.warning(f"Auto-train failed: {exc}")
            return
        try:
            with open(CAT_MODEL_PATH, 'rb') as f: self._cat_pipeline = pickle.load(f)
            with open(PRI_MODEL_PATH, 'rb') as f: self._pri_pipeline = pickle.load(f)
            with open(META_SAVE_PATH, 'rb') as f: self._meta         = pickle.load(f)
            self._trained = True
            logger.info(f"Loaded classifier models (trained {self._meta.get('trained_at', 'unknown')})")
        except Exception as exc:
            logger.warning(f"Could not load classifier: {exc}")
