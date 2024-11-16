import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class AudioPreprocessor:
    def __init__(self, 
                 source_dir='Original Recordings',
                 output_dir='Processed_BirdNET',
                 target_sr=48000,  # BirdNET's expected sample rate
                 duration=3.0,     # Duration in seconds
                 min_duration=  # Minimum duration to keep
                ):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.target_sr = target_sr
        self.duration = duration
        self.min_duration = min_duration
        self.metadata = []
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_audio(self, audio_path):
        """Process a single audio file according to BirdNET specifications"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.target_sr)
            
            # Check duration
            duration = librosa.get_duration(y=y, sr=sr)
            if duration < self.min_duration:
                return None
            
            # Trim or pad to target duration
            target_length = int(self.duration * self.target_sr)
            if len(y) > target_length:
                # Take center portion
                start = (len(y) - target_length) // 2
                y = y[start:start + target_length]
            else:
                # Pad with zeros
                pad_length = target_length - len(y)
                pad_left = pad_length // 2
                pad_right = pad_length - pad_left
                y = np.pad(y, (pad_left, pad_right))
            
            # Create melspectrogram according to BirdNET specifications
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=self.target_sr,
                n_fft=2048,
                hop_length=1024,
                n_mels=128,
                fmin=0,
                fmax=24000
            )
            
            # Convert to log scale
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            return mel_spec
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None
    
    def process_dataset(self):
        """Process entire dataset and create metadata"""
        processed_files = []
        
        # Get all audio files
        audio_files = list(self.source_dir.glob('**/*.mp3')) + list(self.source_dir.glob('**/*.wav'))
        
        for audio_path in tqdm(audio_files, desc="Processing audio files"):
            # Extract class name from directory structure
            species = audio_path.parent.name
            
            # Process audio
            mel_spec = self.process_audio(audio_path)
            if mel_spec is None:
                continue
                
            # Create output filename
            output_filename = f"{species}_{audio_path.stem}.npy"
            output_path = self.output_dir / species / output_filename
            
            # Create species directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save processed spectrogram
            np.save(output_path, mel_spec)
            
            # Add to metadata
            processed_files.append({
                'filename': output_filename,
                'species': species,
                'original_path': str(audio_path),
                'processed_path': str(output_path)
            })
        
        # Create and save metadata DataFrame
        self.metadata = pd.DataFrame(processed_files)
        self.metadata.to_csv(self.output_dir / 'metadata.csv', index=False)
        
        return self.metadata

class BirdNETDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        """
        Args:
            csv_path: Path to your preprocessed_data.csv
            transform: Optional transform to be applied on a sample
        """
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        
        # Create class mapping from species
        self.classes = sorted(self.data['species'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Print some useful information
        print(f"Found {len(self.classes)} unique species")
        print(f"Total number of samples: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load and process audio file
        # Adjust the path construction according to your file structure
        audio_path = os.path.join('Original Recordings', row['file_name'])
        
        try:
            # Load audio with BirdNET specifications
            y, sr = librosa.load(audio_path, sr=48000)  # BirdNET uses 48kHz
            
            # Create melspectrogram according to BirdNET specifications
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=2048,
                hop_length=1024,
                n_mels=128,
                fmin=0,
                fmax=24000
            )
            
            # Convert to log scale
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Convert to tensor and add channel dimension
            mel_spec = torch.FloatTensor(mel_spec).unsqueeze(0)
            
            # Normalize
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
            
            # Get label
            label = self.class_to_idx[row['species']]
            
            if self.transform:
                mel_spec = self.transform(mel_spec)
                
            return mel_spec, label
            
        except Exception as e:
            print(f"Error loading file {audio_path}: {str(e)}")
            # Return a dummy tensor in case of error
            return torch.zeros((1, 128, 128)), -1

def create_dataloaders(csv_path, batch_size=32, train_split=0.8, val_split=0.1):
    """Create train, validation, and test dataloaders"""
    # Create dataset
    dataset = BirdNETDataset(csv_path)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Modify if you want to use multiple workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        num_workers=0
    )
    
    print(f"\nDataset splits:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

# Usage
def main():
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        'preprocessed_data.csv',
        batch_size=32
    )
    
    # Print first batch shape to verify
    for batch, labels in train_loader:
        print(f"\nBatch shape: {batch.shape}")
        print(f"Labels shape: {labels.shape}")
        break
        
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = main()