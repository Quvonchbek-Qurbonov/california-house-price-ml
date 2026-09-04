from fastapi import APIRouter, Query, HTTPException

router = APIRouter(tags=["classification"])


@router.get("/dataset/info", status_code=200)
def get_metrics():
    return {"hello"}