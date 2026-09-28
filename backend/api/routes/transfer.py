"""API routes for Transfer Value Predictor module."""

from fastapi import APIRouter
from backend.api.schemas.transfer import (
    TransferPredictRequest,
    TransferPredictResponse,
    TransferModelInfoResponse,
)
from backend.services.transfer_service import (
    predict_transfer_value_service,
    get_transfer_model_info_service,
)

router = APIRouter(prefix="/transfer", tags=["Transfer Value Predictor"])


@router.post(
    "/predict",
    response_model=TransferPredictResponse,
    summary="Predict Player Transfer Valuation",
    description="Predicts continuous player transfer valuation in EUR from demographic, contractual, and seasonal performance indicators.",
)
def predict_transfer_value_endpoint(request: TransferPredictRequest) -> TransferPredictResponse:
    """Generate estimated player market value in EUR."""
    return predict_transfer_value_service(request)


@router.get(
    "/model-info",
    response_model=TransferModelInfoResponse,
    summary="Transfer Value Model Metadata & Performance",
    description="Returns training dataset statistics, feature list, and evaluation metrics (MAE, RMSE, R²) for the transfer model.",
)
def get_transfer_model_info_endpoint() -> TransferModelInfoResponse:
    """Retrieve transfer value model training metadata and validation scores."""
    return get_transfer_model_info_service()


@router.get(
    "/presets",
    summary="Dynamic Current Player Presets",
    description="Returns dynamically generated presets for current stars (Cherki, Haaland, Saka, Rice, Van Dijk, Martinez) populated with latest 2025/26 features.",
)
def get_transfer_presets_endpoint():
    """Retrieve dynamic presets populated with current data."""
    from backend.services.transfer_service import get_transfer_presets_service
    return get_transfer_presets_service()


@router.get(
    "/player-features/{player_name}",
    summary="Get Current Player Transfer Features",
    description="Looks up the complete 18-feature input vector for a current player from data/current/players/current_transfer_features.csv.",
)
def get_player_transfer_features_endpoint(player_name: str):
    """Retrieve current player input features for transfer valuation."""
    from backend.services.transfer_service import get_player_transfer_features_service
    return get_player_transfer_features_service(player_name)

