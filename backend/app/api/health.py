from fastapi import APIRouter

router = APIRouter(prefix="/v1", tags=["health"])


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "nexum-shield"
    }