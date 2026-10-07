"""
Cloud MLOps Deployment Utilities for AWS SageMaker and Google Cloud Vertex AI.
Demonstrates enterprise operationalization patterns for production serving.
"""
import argparse


def deploy_to_aws_sagemaker(
    model_artifact_path: str = "models/customer_churn_model.pkl",
    role_arn: str = "arn:aws:iam::123456789012:role/SageMakerExecutionRole",
    instance_type: str = "ml.m5.large",
    endpoint_name: str = "customer-churn-endpoint",
):
    """
    Template for deploying trained Scikit-learn model to Amazon SageMaker Real-Time Endpoint.
    Uses SageMaker Python SDK (boto3 / sagemaker.sklearn.model.SKLearnModel).
    """
    print(f"[AWS SageMaker] Preparing model deployment to endpoint: {endpoint_name}")
    print(f"  Artifact: {model_artifact_path}")
    print(f"  Instance Type: {instance_type}")
    print(f"  Execution Role: {role_arn}")

    sagemaker_code = f'''
    from sagemaker.sklearn.model import SKLearnModel

    model = SKLearnModel(
        model_data="s3://my-mlops-bucket/models/customer_churn_model.tar.gz",
        role="{role_arn}",
        entry_point="app/sagemaker_entrypoint.py",
        framework_version="1.2-1",
        py_version="py3",
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="{instance_type}",
        endpoint_name="{endpoint_name}",
    )
    '''
    print("[AWS SageMaker] Deployment definition generated successfully.")
    return sagemaker_code


def deploy_to_gcp_vertex_ai(
    model_artifact_path: str = "models/customer_churn_model.pkl",
    project_id: str = "my-gcp-project",
    region: str = "us-central1",
    endpoint_display_name: str = "customer-churn-vertex-endpoint",
):
    """
    Template for deploying containerized model to Google Cloud Vertex AI Endpoint.
    Uses Google Cloud AI Platform SDK (google.cloud.aiplatform).
    """
    print(f"[GCP Vertex AI] Preparing model deployment in region: {region}")
    print(f"  Project: {project_id}")
    print(f"  Display Name: {endpoint_display_name}")

    vertex_code = f'''
    from google.cloud import aiplatform

    aiplatform.init(project="{project_id}", location="{region}")

    # Register custom container in Vertex AI Model Registry
    model = aiplatform.Model.upload(
        display_name="{endpoint_display_name}",
        artifact_uri="gs://my-churn-bucket/model-artifacts/",
        serving_container_image_uri="gcr.io/{project_id}/churn-mlops:latest",
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_ports=[8000],
    )

    # Deploy model to real-time endpoint
    endpoint = model.deploy(
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=3,
        traffic_percentage=100,
    )
    '''
    print("[GCP Vertex AI] Deployment definition generated successfully.")
    return vertex_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloud MLOps Deployment Utility")
    parser.add_argument(
        "--platform",
        choices=["sagemaker", "vertex"],
        default="sagemaker",
        help="Cloud platform target",
    )
    args = parser.parse_args()

    if args.platform == "sagemaker":
        deploy_to_aws_sagemaker()
    else:
        deploy_to_gcp_vertex_ai()
