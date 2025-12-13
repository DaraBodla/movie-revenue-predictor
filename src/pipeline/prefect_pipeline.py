from prefect_pipeline import ml_training_pipeline

if __name__ == '__main__':
    result = ml_training_pipeline()
    print(f"\nPipeline Result: {result}")
