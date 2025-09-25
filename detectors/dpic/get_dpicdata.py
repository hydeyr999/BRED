import pandas as pd

domains = ['xsum','pubmedqa','squad','writingprompts','openreview','blog','tweets']
models = ['llama', 'ds', '4o','Qwen']
operations = ['create','rewrite','summary','polish','refine','expand','translate']

def get_domain(domain):
    text = []
    generated_text = []
    generated = []
    for model in models:
        for operation in operations:
            data_path = f'./data_gen/LLM-texts-new/{operation}/{domain}_{model}_{operation}_dpic.csv'
            df = pd.read_json(data_path,lines=True)

            text.extend(df['text'].tolist())
            generated_text.extend(df['dpic_text'].tolist())
            generated.extend(df['generated'].tolist())

    df = pd.DataFrame({
        'id': range(len(text)),
        'text': text,
        'generated_text': generated_text,
        'generated': generated,
    })
    print(df)

    df.to_json(f'./detectors/DPIC/dpic_data/cross-domain/{domain}_sample_dpic.json', orient='records',lines=True)

def get_model(model):
    text = []
    generated_text = []
    generated = []
    for domain in domains:
        for operation in operations:
            data_path = f'./data_gen/LLM-texts-new/{operation}/{domain}_{model}_{operation}_dpic.csv'
            df = pd.read_json(data_path,lines=True)

            text.extend(df['text'].tolist())
            generated_text.extend(df['dpic_text'].tolist())
            generated.extend(df['generated'].tolist())


    df = pd.DataFrame({
        'id': range(len(text)),
        'text': text,
        'generated_text': generated_text,
        'generated': generated,
    })
    print(df)

    df.to_json(f'./detectors/DPIC/dpic_data/cross-model/{model}_sample_dpic.json', orient='records',lines=True)

def get_operation(operation):
    text = []
    generated_text = []
    generated = []
    for domain in domains:
        for model in models:
            data_path = f'./data_gen/LLM-texts-new/{operation}/{domain}_{model}_{operation}_dpic.csv'
            df = pd.read_json(data_path,lines=True)

            text.extend(df['text'].tolist())
            generated_text.extend(df['dpic_text'].tolist())
            generated.extend(df['generated'].tolist())

    df = pd.DataFrame({
        'id': range(len(text)),
        'text': text,
        'generated_text': generated_text,
        'generated': generated,
    })
    print(df)

    df.to_json(f'./detectors/DPIC/dpic_data/cross-operation/{operation}_sample_dpic.json', orient='records',lines=True)


if __name__ == "__main__":

    for domain in domains:
        get_domain(domain)

    for model in models:
        get_model(model)

    for operation in operations:
        get_operation(operation)