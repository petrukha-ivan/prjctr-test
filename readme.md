## General

### **Bootstrap Environment**

Before starting you need to prepare the environment:

```sh
python -m venv venv && source venv/bin/activate
```

To install dependencies:

```sh
pip install -r requirements.txt
```

## Trainer

### **Train Model**

To train model with default parameters simply execute next command:

```sh
python -m trainer
```

### **Custom Arguments**

To specify pretrained model from Hugging Face Hub use arguments:

```sh
python -m trainer --model_checkpoint distilbert/distilbert-base-cased
```

To show all available arguments and adjust training:

```sh
python -m trainer --help
```

## Server

### **Serve Endpoint**

To serve prediction endpoint simply execute next command:

```sh
python -m server --model_checkpoint distilbert/distilbert-base-cased
```

### **Request Endpoint**

cURL request example:

```sh
curl -X "POST" "http://127.0.0.1:8000/score" \
     -H 'Content-Type: application/json; charset=utf-8' \
     -d $'{"text": "Is this text easy to read?"}'
```