"""
dataset.py - Deterministic Loan Applications Dataset Generator
Track: Banking & FinTech (Cred)
Task 1: Seeded dataset generator with invariant validation.
"""

from typing import List, Dict, Any, Optional
import random

# Controlled Vocabularies
LOAN_CATEGORIES = [
    "Personal Loan",
    "Home Loan",
    "Auto Loan",
    "Education Loan",
    "Business Loan",
]

APPLICATION_STATUSES = [
    "Submitted",
    "Under Review",
    "Approved",
    "Rejected",
    "Disbursed",
]

# Lending Amount Bounds (INR) based on product guidelines
LENDING_BOUNDS: Dict[str, tuple] = {
    "Personal Loan": (50_000, 1_500_000),
    "Home Loan": (1_500_000, 25_000_000),
    "Auto Loan": (200_000, 3_500_000),
    "Education Loan": (100_000, 5_000_000),
    "Business Loan": (500_000, 10_000_000),
}


class LoanApplicationRecord:
    """Represents a single immutable customer loan application record."""

    def __init__(
        self,
        record_id: str,
        category: str,
        status: str,
        loan_amount_inr: float,
        days_since_created: int,
        flagged_for_fraud_review: bool,
    ):
        self.record_id = record_id
        self.category = category
        self.status = status
        self.loan_amount_inr = round(float(loan_amount_inr), 2)
        self.days_since_created = int(days_since_created)
        self.flagged_for_fraud_review = bool(flagged_for_fraud_review)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "category": self.category,
            "status": self.status,
            "loan_amount_inr": self.loan_amount_inr,
            "days_since_created": self.days_since_created,
            "flagged_for_fraud_review": self.flagged_for_fraud_review,
        }

    def __repr__(self) -> str:
        return f"LoanApplicationRecord({self.record_id}, {self.category}, {self.status}, ₹{self.loan_amount_inr:,.2f})"


def generate_loan_applications(
    seed: int = 42, total_records: int = 45, target_fraud_prob: float = 0.20
) -> List[Dict[str, Any]]:
    """
    Deterministically generates loan application records meeting all capstone invariants.
    Invariant 1: Total records >= 40.
    Invariant 2: Every category >= 3 records.
    Invariant 3: Every status >= 1 record.
    Invariant 4: Fraud review proportion in [0.10, 0.30].
    """
    if total_records < 40:
        raise ValueError("Total records must be >= 40.")

    rng = random.Random(seed)
    records: List[Dict[str, Any]] = []

    # Guarantee baseline coverage: all categories >= 3, all statuses >= 1
    # 5 categories * 3 = 15 guaranteed baseline records
    baseline_records: List[Dict[str, Any]] = []
    rec_counter = 1001

    for cat_idx, category in enumerate(LOAN_CATEGORIES):
        min_bound, max_bound = LENDING_BOUNDS[category]
        for repeat_idx in range(3):
            # Assign status cycling through APPLICATION_STATUSES to guarantee coverage
            status = APPLICATION_STATUSES[(cat_idx * 3 + repeat_idx) % len(APPLICATION_STATUSES)]
            amount = rng.uniform(min_bound, max_bound)
            days = rng.randint(0, 30)
            is_fraud = rng.random() < target_fraud_prob

            rec = LoanApplicationRecord(
                record_id=f"APP-{rec_counter}",
                category=category,
                status=status,
                loan_amount_inr=amount,
                days_since_created=days,
                flagged_for_fraud_review=is_fraud,
            )
            baseline_records.append(rec.to_dict())
            rec_counter += 1

    records.extend(baseline_records)

    # Fill remaining records up to total_records
    remaining = total_records - len(records)
    for _ in range(remaining):
        category = rng.choice(LOAN_CATEGORIES)
        status = rng.choice(APPLICATION_STATUSES)
        min_bound, max_bound = LENDING_BOUNDS[category]
        amount = rng.uniform(min_bound, max_bound)
        days = rng.randint(0, 30)
        is_fraud = rng.random() < target_fraud_prob

        rec = LoanApplicationRecord(
            record_id=f"APP-{rec_counter}",
            category=category,
            status=status,
            loan_amount_inr=amount,
            days_since_created=days,
            flagged_for_fraud_review=is_fraud,
        )
        records.append(rec.to_dict())
        rec_counter += 1

    # Validate and programmatically calibrate fraud review band constraint [0.10, 0.30]
    fraud_count = sum(1 for r in records if r["flagged_for_fraud_review"])
    fraud_rate = fraud_count / len(records)

    if not (0.10 <= fraud_rate <= 0.30):
        # Adjust target fraud probability and recursively regenerate
        adjusted_target = min(max(0.15, target_fraud_prob), 0.25)
        return generate_loan_applications(seed=seed + 1, total_records=total_records, target_fraud_prob=adjusted_target)

    # Invariant verification assertions
    for cat in LOAN_CATEGORIES:
        count = sum(1 for r in records if r["category"] == cat)
        assert count >= 3, f"Invariant violated: category {cat} has only {count} records."

    for st in APPLICATION_STATUSES:
        count = sum(1 for r in records if r["status"] == st)
        assert count >= 1, f"Invariant violated: status {st} has only {count} records."

    assert 0.10 <= fraud_rate <= 0.30, f"Invariant violated: fraud rate {fraud_rate} outside [0.10, 0.30]."

    return records


# Seeded production dataset instance
LOAN_APPLICATIONS: List[Dict[str, Any]] = generate_loan_applications(seed=42, total_records=45)

# Fast index lookup dictionary by record_id
LOAN_APPLICATIONS_BY_ID: Dict[str, Dict[str, Any]] = {
    r["record_id"]: r for r in LOAN_APPLICATIONS
}


def get_loan_application(record_id: str) -> Optional[Dict[str, Any]]:
    """O(1) transactional lookup for loan application records."""
    normalized_id = record_id.strip().upper()
    return LOAN_APPLICATIONS_BY_ID.get(normalized_id)


if __name__ == "__main__":
    print(f"Generated {len(LOAN_APPLICATIONS)} loan applications.")
    fraud_count = sum(1 for r in LOAN_APPLICATIONS if r["flagged_for_fraud_review"])
    print(f"Fraud flagged: {fraud_count}/{len(LOAN_APPLICATIONS)} ({fraud_count/len(LOAN_APPLICATIONS):.1%})")
    print("Sample record:", LOAN_APPLICATIONS[0])
