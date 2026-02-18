# Pathogenic-Gene-Prediction
This project implements a deep learning model designed to predict whether a human genetic variant is pathogenic or benign based on raw DNA sequences.  

Interpreting genetic variants remains a major bottleneck in genomics and precision medicine. This project explores how machine learning can assist in prioritizing variants for further biological or clinical investigation.

*Objectives*
1. Build a model capable of classifying variants by pathogenicity
2. Learn sequence patterns associated with harmful mutations
3. Create a reproducible pipeline for genomic data preprocessing and modeling

*Dataset*
Variant labels were obtained from ClinVar, a public archive of human genetic variants and their clinical interpretations. From ClinVar, the dataset was filtered to include variants with clear Benign or Pathogenic labels, single nucleotide variants (SNVs) and entries without conflicting clinical interpretations. Fields used include chromosome, position, reference allele, alternate allele, and clinical significance.

*Reference Genome*
Sequence context for each variant was extracted from the human reference genome obtained via NCBI. The reference genome enables extraction of nucleotide windows surrounding each variant and the construction of sequence-based model inputs.

*Model Architecture*
The model is a 1D convolutional neural network (CNN) designed to learn sequence patterns associated with pathogenic mutations from one-hot encoded DNA sequences. It consists of two convolutional layers for feature extraction and two dense layers for classification. 

*Results*
Accuracy: 57%

*Future work*
1. Incorporate structural variants and indels
2. Integrate conservation and functional annotation features
3. Explore transformer-based genomic models
