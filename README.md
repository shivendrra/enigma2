# Enigma2

DNA sequence predition using Transformer-based models. Second version of [Enigma](https://github.com/shivendrra/enigma). It has two models- one BERT-based model for classification & analysis, other transformer-based alphafold like model.

## Sepcifications

More technical details about models is available in the documentation: [Models.md](https://github.com/shivendrra/enigma2/blob/main/docs/Model.md)

## Database

Utilizes a custom-built pipeline to fetch datasets from trust NCBI database using [EnigmaDataset](https://github.com/delveopers/EnigmaDataset) library, that could be downloaded and used by anyone with proper NCBI-specified parameters. If you want to download pre-fetched database, you can dowload it from here- [huggingface/EnigmaDatasaet](https://huggingface.co/datasets/shivendrra/EnigmaDataset)

## Contributing

1. Fork the repository.

2. Create a feature branch:
  
  ```bash
  git checkout -b feature-name
  ```

3. Commit your changes:

  ```bash
  git commit -m "Add feature"
  ```

4. Push to the branch:

  ```bash
  git push origin feature-name
  ```

5. Create a pull request.

## License

This project is licensed under the Apache 2 License. See the [LICENSE](LICENSE) file for details.
