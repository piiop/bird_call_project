# Bird Call Classification Project
A machine learning system for identifying Ohio bird species through audio calls.

## Project Overview
This project implements multiple machine learning approaches to classify bird species from their calls, focusing on birds native to Ohio. It explores both traditional ML methods using extracted audio features and deep learning approaches using spectrograms. The project progresses from basic ML models to more advanced CNN architectures, serving as both a practical bird classification system and a learning journey in audio processing and machine learning.

Key Features:
- Multi-model approach comparing traditional ML vs deep learning
- Focus on 121 Ohio bird species
- Audio processing pipeline from raw calls to features/spectrograms
- Experimental comparison of different classification approaches

## Project Goals
1. Create a classification system achieving >X% accuracy across 121 Ohio bird species
2. Implement and compare multiple approaches:
   - Traditional ML models using extracted audio features
   - CNN models using spectrogram images
   - Fine-tuned pretrained models
3. Develop a modular, maintainable codebase following software engineering best practices
4. Create a simple demo interface using Gradio for practical testing

## Dataset
### Source
- Data collected via [Xeno-canto](https://xeno-canto.org/) API
- 756 initial recordings of birds from Ohio
- 127 unique species (64 unidentified recordings reserved for future use)
- No duplicate recordings present in dataset


Field Description:
genus          : Bird genus (692 entries)
species        : Species name (692 entries)
latitude       : Recording location latitude (692 entries)
longitude      : Recording location longitude (692 entries)
quality        : Recording quality A-E (692 entries)
file_name      : Audio file identifier (692 entries)
simplified_type: Song/Call/Other/Unknown (692 entries)
season         : Recording season (692 entries)
time_of_day    : Recording time period (692 entries)
length_seconds : Duration in seconds (692 entries)

### Recording Statistics
- Duration range: 1-576 seconds
- Quality ratings: A through E
- Geographic coverage: Ohio, USA (latitude/longitude coordinates)

### Metadata Processing
Initial dataset (756 entries, 16 fields) was processed to create a cleaned dataset (692 entries, 10 fields) through the following steps:

#### Fields Retained and Processed:
- **genus**: Bird genus
- **species**: Species name
- **latitude/longitude**: Recording location coordinates
- **quality**: Recording quality (A-E)
- **file_name**: Audio file identifier
- **type**: Vocalization type
 - Simplified from 44 categories to ['Song', 'Call', 'Other', 'Unknown']
- **date/time**: Processed into two new fields:
 - **season**: Derived from recording date
 - **time_of_day**: Derived from recording time
- **length**: Standardized to **length_seconds** in integer format

#### Fields Not Used (Reserved for Future Analysis):
- **remarks**: Additional notes about recordings
- **also**: Presence of other species in recordings
- **sex**: Bird sex identification (dropped due to missing data)
- **stage**: Life stage of bird (dropped due to missing data and adult bias)

### Final Dataset Structure
```python
Dataset Overview:
- 692 entries
- 10 metadata fields
- Memory usage: 59.5+ KB

## Data Organization
```python
data/
├── original_recordings/    # Raw audio files from Xeno-canto
├── augmented_recordings/   # Audio files with data augmentation applied
├── converted_recordings/   # Audio files converted to standardized format
├── processed_recordings/   # Fully preprocessed audio ready for feature extraction
└── metadata/              # CSV files containing recording information and labels
```


## Project Steps

### 1. Data Collection
#### Data Source
- Collected 756 recordings via Xeno-canto API, targeting birds found in Ohio
- Downloaded raw audio files and comprehensive metadata
- Implemented systematic API querying to ensure complete species coverage

#### Initial Data Processing
- Verified data integrity with no duplicate recordings
- Cleaned and standardized metadata fields:
  - Consolidated 44 vocalization types into 4 categories (Song/Call/Other/Unknown)
  - Created derivative fields (season, time_of_day) from timestamp data
  - Standardized duration measurements to seconds
- Final cleaned dataset: 692 recordings across 127 species

#### Quality Control
- Retained all quality ratings (A-E) for analysis
- Reserved 64 unidentified recordings for potential future use
- Maintained complete geographic data (latitude/longitude) for all recordings

### 2. Exploratory Data Analysis (EDA)
#### Dataset Composition
- Analyzed distribution of 692 recordings across 127 species
- Evaluated recording lengths (range: 1-576 seconds)
- Mapped geographic distribution of recordings across Ohio
[Insert Visual]
#### Metadata Analysis
- Vocalization Types
  - Distribution across simplified categories (Song/Call/Other/Unknown)
  - Analysis of original 44 vocalization categories before simplification
- Temporal Patterns
  - Seasonal distribution of recordings
  - Time of day recording patterns
- Quality Assessment
  - Distribution of quality ratings (A-E)
  - Relationship between quality and recording length

#### Visualizations
- Recording length distribution plots
- Species frequency distribution
- Seasonal and daily recording patterns
- Quality rating distributions

#### Key Insights
- Identified potential class imbalances across species
- Mapped temporal patterns in recording availability
- Analyzed geographic coverage and potential sampling biases

### 3. Audio Processing
#### Initial Quality Assessment
- Manual review of low-quality recordings
 - Evaluated 8 files with quality rating 'E' or 'no score'
 - Determined all files were usable for analysis
 - Retained all recordings for further processing

#### Audio Standardization
- Converted all recordings from MP3 to WAV format using librosa
- Standardized sample rates across dataset
 - Analyzed 690 files
 - Resampled 311 files to consistent rate
 - Maintained original sample rate for 379 files

#### Audio Cleaning Pipeline
Implemented comprehensive audio quality control:

1. **Noise Analysis**
  - Calculated Signal-to-Noise Ratio (SNR) for each recording
  - Filtered recordings below SNR threshold (-20 dB)

2. **Silence Detection**
  - Identified prolonged silence periods (threshold: -60 dB)
  - Flagged recordings with silence exceeding 1.0 second

3. **Spectral Analysis**
  - Computed spectral spread for each recording
  - Filtered recordings exceeding spread threshold (0.8)

4. **Duplicate Detection**
  - Generated MFCC-based audio fingerprints
  - Compared fingerprints using similarity threshold (0.99)
  - Confirmed no duplicate recordings in dataset

#### Audio Segmentation
Standardized recording lengths for consistent model input:
- Minimum length threshold: 100ms
- Target segment length: 5 seconds
- Processing method:
 - Short recordings: Zero-padded to target length
 - Long recordings: Segmented with overlap
 - Each segment maintains original audio characteristics

 #### Data Augmentation
Applied systematic augmentation to expand dataset and improve model robustness:

1. **Augmentation Techniques**
   - Pitch shifting (±2 steps)
   - Time stretching (rate: 1.2)
   - Noise addition (factor: 0.01)
   - Speed modification (factor: 1.1)
   - Frequency filtering (lowpass: 2000Hz cutoff)
   - Time shifting (max: 0.5 seconds)
   - Nature sound mixing (matched to original duration)

2. **Dataset Growth**
   - Initial processed dataset: 12,120 segments
     - Derived from 691 original recordings
     - Each recording split into 5-second segments
   - Final augmented dataset: 48,174 segments
     - Applied multiple augmentation techniques per segment
     - ~4x increase in dataset size

3. **Data Tracking**
   - Added 'augmentations' column to metadata
   - Maintained all original metadata fields
   - Preserved relationship between original and augmented samples

#### Dataset Evolution
1. Original collection: 756 recordings
2. After initial cleaning: 691 recordings
3. After segmentation: 12,120 segments
4. After augmentation: 48,174 total samples

Memory usage increased from 1.1+ MB to 4.4+ MB with maintained data integrity across all fields.

#### Quality Control Pipeline
1. Single-file testing phase
  - Validated each processing step on sample recordings
  - Adjusted thresholds based on manual verification
2. Batch processing
  - Applied validated pipeline to entire dataset
  - Logged all modifications and filtering decisions

### 4. Feature Extraction and Dataset Preparation
#### Audio Feature Extraction
Extracted comprehensive acoustic features from each 5-second segment:

1. **Mel-frequency Cepstral Coefficients (MFCCs)**
  - Primary features capturing timbral characteristics
  - Represents the spectral envelope of the audio signal

2. **Spectral Features**
  - Spectral Centroids: Indicating the "center of mass" of the spectrum
  - Spectral Rolloff: Frequency below which 85% of spectral energy lies
  - Chroma Features: Representation of pitch content

3. **Time Domain Features**
  - Zero Crossing Rate: Indicating frequency of signal sign-changes
  - Useful for discriminating between voiced and unvoiced sounds

#### Dataset Organization
Created two parallel datasets for model comparison:

1. **Tabular Dataset**
  - Format: Feature vectors stored in numpy arrays
  - Size: 48,174 samples × N features
  - Each row corresponds to a processed and augmented 5-second segment
  - Features unstacked for traditional ML model input
  
2. **Spectrogram Dataset**
  - Format: Image representations of audio segments
  - Generated from processed and augmented recordings
  - Maintained 1:1 correspondence with tabular dataset
  - Structured for CNN model input

#### Data Storage
- Feature vectors stored in 'feature_vector' column of main DataFrame
- Maintained links between features and original metadata
- Preserved augmentation information for traceability

### 5. Model Development
- Train and evaluate multiple model types, including but not limited to:
  - Classical machine learning models (e.g., Random Forests, SVM)
  - Deep learning models (e.g., CNNs, RNNs)
- Compare model performances across different architectures and datasets

### 6. Model Evaluation
- Assess model accuracy, precision, recall, and F1-score
- Analyze confusion matrices to identify challenging species
- Compare performance between tabular and spectrogram-based models

## Technologies Used
- Python
- Libraries: (list key libraries used, e.g., TensorFlow, PyTorch, librosa, pandas, scikit-learn)

## Setup and Installation
(Provide instructions for setting up the project environment)

## Usage
(Explain how to run the project, including any command-line instructions)

## Results
(Summarize key findings and model performance)

## Future Work
- Expand the dataset to include more species or geographic regions
- Implement real-time bird call identification
- Develop a user-friendly interface for model predictions

## Contributing
This is a personal project, but suggestions and feedback are welcome. Please open an issue to discuss any proposed changes.

## License
(Specify the license under which you're releasing this project)

## Acknowledgments
- Xeno-canto for providing the bird call recordings
- (Any other resources or individuals you'd like to acknowledge)