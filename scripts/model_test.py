# general libraries
import os

# hugging face libraries
from transformers import (
    AutoFeatureExtractor,
    AutoModelForImageClassification, 
    TrainingArguments, 
    Trainer
)

# pytorch libraries
import torch
from torchvision.transforms import (
    Compose, 
    Normalize,
    Resize,
    ToTensor
)


def model_test(path, dataset):
    # Load the pre-trained model and feature extractor
    model_path = os.path.join(path, 'saved_model')
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    model = AutoModelForImageClassification.from_pretrained(model_path)
     
    # Define the normalization transformation
    normalize = Normalize(
        mean=feature_extractor.image_mean, 
        std=feature_extractor.image_std
    )
    
    # Define the test transformation pipeline
    test_transform = Compose([
        Resize(size=224),
        ToTensor(),
        normalize
    ])

    # Preprocess function to apply transformations to the test dataset
    def preprocess(example_batch):
        example_batch["pixel_values"] = [test_transform(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    # Prepare the test dataset and apply the preprocess function
    test_set = dataset['test']
    test_set.set_transform(preprocess)
    
    # Collate function for the test dataset
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}
    
    # Define the training arguments for the Trainer
    args = TrainingArguments(
        output_dir='./',
        remove_unused_columns=False,
        per_device_eval_batch_size=32,
        push_to_hub=False,
        report_to='none'
    )
    
    # Create the Trainer instance
    tester = Trainer(
        model=model,
        args=args,
        eval_dataset=test_set,
        tokenizer=feature_extractor,
        data_collator=collate_fn
    )
    
    # Make predictions on the test dataset using the Trainer
    predictions = tester.predict(test_dataset=test_set).predictions
    
    return predictions