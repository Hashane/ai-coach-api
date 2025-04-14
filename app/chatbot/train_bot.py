import click
from trainer import train_embeddings

@click.command()
@click.option('--data', default='data/data.json', help='Path to the intents JSON file')
@click.option('--embeddings', default='data/sbert_embeddings.pkl', help='Path to save embeddings')
@click.option('--labels', default='data/label_encoder.pkl', help='Path to save label encoder')
@click.option('--responses', default='data/responses_dict.pkl', help='Path to save responses dictionary')
@click.option('--model', default='jgammack/distilbert-base-mean-pooling', help='HuggingFace model to use')
@click.option('--model_dir', default='/Users/judehashane/PycharmProjects/AICoachApiProject/app/chatbot', help='Working directory for model and data')
def cli(data, embeddings, labels, responses, model, model_dir):
    train_embeddings(
        data_path=data,
        embedding_output=embeddings,
        label_output=labels,
        responses_output=responses,
        model_name=model,
        model_dir=model_dir
    )

if __name__ == '__main__':
    cli()
