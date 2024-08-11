This folder has the container related to FAISS & Vector Encoding of images using VIT MAE.

"FAISS (Facebook AI Similarity Search) is a library that allows developers to quickly search for embeddings of multimedia documents that are similar to each other. It solves limitations of traditional query search engines that are optimized for hash-based searches, and provides more scalable similarity search functions."

https://ai.meta.com/tools/faiss/

"The ViTMAE model was proposed in Masked Autoencoders Are Scalable Vision Learners by Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick.  The paper shows that, by pre-training a Vision Transformer (ViT) to reconstruct pixel values for masked patches, one can get results after fine-tuning that outperform supervised pre-training."

https://arxiv.org/abs/2111.06377v2

https://huggingface.co/docs/transformers/v4.19.2/en/model_doc/vit_mae


To build the container use:
> docker build -t faiss_docker -f Dockerfile .

To run the container use:
> docker run --rm --name faiss_docker -ti faiss_docker

To try sample FAISS code (from FAISS tutorial) use:
> python3 src/faiss_tutorial.py

To try sample FAISS code (from FAISS tutorial) use:
> python3 src/vit_mae_encoder.py