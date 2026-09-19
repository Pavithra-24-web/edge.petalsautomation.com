"""
API v1 router — aggregates all endpoint modules
"""
from fastapi import APIRouter

from app.motion.api import router as motion_router
from app.api.v1.endpoints import (
    auth,
    projects,
    samples,
    labels,
    impulses,
    dsp,
    training,
    evaluation,
    deployment,
    devices,
    inference,
    ai_labeling,
    post_processing,
    post_processing_pipeline,
    trained_models,
    model_testing,
    ingestion,
    device_packages,
    device_catalog,
    demo_requests,
    contact_requests,
    project_versions,
    synthetic_data,
)

router = APIRouter()

router.include_router(auth.router,            prefix="/auth",            tags=["auth"])
router.include_router(projects.router,        prefix="/projects",        tags=["projects"])
router.include_router(samples.router,         prefix="/samples",         tags=["samples"])
router.include_router(labels.router,          prefix="/labels",          tags=["labels"])
router.include_router(impulses.router,        prefix="/impulses",        tags=["impulses"])
router.include_router(dsp.router,             prefix="/dsp",             tags=["dsp"])
router.include_router(training.router,        prefix="/training",        tags=["training"])
router.include_router(evaluation.router,      prefix="/evaluation",      tags=["evaluation"])
router.include_router(deployment.router,      prefix="/deployment",      tags=["deployment"])
router.include_router(devices.router,         prefix="/devices",         tags=["devices"])
router.include_router(motion_router,          prefix="/devices",         tags=["motion"])
router.include_router(inference.router,       prefix="/inference",       tags=["inference"])
router.include_router(ai_labeling.router,     prefix="/ai-labeling",     tags=["ai-labeling"])
router.include_router(post_processing.router, prefix="/post-processing", tags=["post-processing"])
router.include_router(post_processing_pipeline.router, prefix="/projects", tags=["post-processing-pipeline"])
router.include_router(trained_models.router,  prefix="/trained-models",  tags=["trained-models"])
router.include_router(model_testing.router,   prefix="/model-testing",   tags=["model-testing"])
router.include_router(ingestion.router,       prefix="/ingestion",       tags=["ingestion"])
router.include_router(device_packages.router, prefix="/device-client",   tags=["device-client"])
router.include_router(device_catalog.router,  prefix="/device-catalog",  tags=["device-catalog"])
router.include_router(demo_requests.router,   prefix="",                 tags=["demo-requests"])
router.include_router(contact_requests.router, prefix="",                tags=["contact-requests"])
router.include_router(project_versions.router, prefix="/project-versions", tags=["project-versions"])
router.include_router(synthetic_data.router,  prefix="/synthetic-data",  tags=["synthetic-data"])
