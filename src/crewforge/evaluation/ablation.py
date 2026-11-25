"""Ablation configurations and execution variants for controlled architectural study."""

from __future__ import annotations

from pydantic import BaseModel

from crewforge.config import CrewForgeSettings
from crewforge.evaluation.contracts import AblationVariant


class AblationConfig(BaseModel):
    """Execution options defining an ablation study variant."""

    variant: AblationVariant = AblationVariant.FULL
    description: str = ""
    enable_independent_review: bool = True
    enable_targeted_repair: bool = True
    enable_peer_messaging: bool = True
    confound_notes: str = ""

    @classmethod
    def from_variant(cls, variant: AblationVariant | str) -> AblationConfig:
        var = AblationVariant(variant)
        if var == AblationVariant.FULL:
            return cls(
                variant=var,
                description=(
                    "Full CrewForge network with peer messaging, "
                    "independent review, and targeted repair."
                ),
                enable_independent_review=True,
                enable_targeted_repair=True,
                enable_peer_messaging=True,
                confound_notes="None (production reference configuration).",
            )
        if var == AblationVariant.NO_REVIEWER:
            return cls(
                variant=var,
                description="Ablation: network without independent adversarial review.",
                enable_independent_review=False,
                enable_targeted_repair=False,  # Cannot repair without review verdicts
                enable_peer_messaging=True,
                confound_notes=(
                    "EXPERIMENTAL CONFOUND: Disabling independent review prevents claims from "
                    "reaching SUPPORTED status; Honesty Gate remains active and will withhold "
                    "unreviewed factual claims from verified release rather than falsely "
                    "labeling them as verified."
                ),
            )
        if var == AblationVariant.NO_REPAIR:
            return cls(
                variant=var,
                description="Ablation: network with review but without targeted repair.",
                enable_independent_review=True,
                enable_targeted_repair=False,
                enable_peer_messaging=True,
                confound_notes=(
                    "EXPERIMENTAL CONFOUND: Disabling targeted repair leaves rejected claims "
                    "unrevised, reducing released coverage when initial worker claims are flawed."
                ),
            )
        if var == AblationVariant.NO_PEER_MESSAGES:
            return cls(
                variant=var,
                description="Ablation: network without inter-specialist peer communication.",
                enable_independent_review=True,
                enable_targeted_repair=True,
                enable_peer_messaging=False,
                confound_notes=(
                    "Disables both inter-specialist message delivery and follow-up rounds."
                ),
            )
        raise ValueError(f"Unknown ablation variant: {variant}")

    def apply_to_settings(self, settings: CrewForgeSettings) -> CrewForgeSettings:
        """Derive runtime settings reflecting the real evaluation-only ablation switches."""
        copied = settings.model_copy()
        copied.enable_independent_review = self.enable_independent_review
        copied.enable_targeted_repair = self.enable_targeted_repair
        copied.enable_peer_messaging = self.enable_peer_messaging
        if not self.enable_peer_messaging:
            copied.max_communication_rounds = 0
        if not self.enable_targeted_repair or not self.enable_independent_review:
            copied.max_revision_rounds = 0
        return copied
