from fastapi import APIRouter, Query, HTTPException
from app.services.clustering import apply_clustering
import timeit

router = APIRouter(tags=["classification"])


@router.get("/dataset/info", status_code=200)
def get_metrics(algorithm: str):
    start_time = timeit.default_timer()
    number_of_mismatches = apply_clustering(algorithm)
    time_taken = timeit.default_timer() - start_time
    return {
        "Number of mismatches": number_of_mismatches,
        "Time taken": time_taken
    }