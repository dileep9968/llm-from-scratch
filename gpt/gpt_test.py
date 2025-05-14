import pytest 
import gpt_train import main
import http.client
from urllib.parse import urlparse

@pytest.fixture
def gpt_config():
    return {
        'vocab_size':50257,
        'context_length':12,
        'emb_dim':32,
        'n_heads':4,
        'n_layers':2,
        'drop_rate':0.1,
        'qkv_bias':False,
    }

@pytest.fixture
def other_settings():
    return {
        'learning_rate':5e-4,
        'num_epochs':1,
        'batch_size':2,
        'weight_decay':0.1
    }

def check_file_size(url, expected_size):
    parsed_url = urlparse(url)
    if parsed_url.scheme =='https':
        conn = http.client.HTTPSConnection(parsed_url.netloc)
    else:
        conn = http.client.HTTPConnection(parsed_url.netloc)

    conn.request('HEAD', parsed_url.path)
    response = conn.getresponse()
    if response.status !=200:
        return False, f"{url} not accessible"
    size = response.getheader('Content-Length')
    if size is None:
        return False, 'Content-Length header is missing'
    if size != expected_size:
        return False, f"{url} file has expected size {expected_size}, but got {size}"
    return True, f"{url} file size is correct"

def test_model_files():
    def check_model_files(base_url):
        model_size = '124M'
        files = {
            'checkpoint':77,
            'encoder.json':1042301,
            'model.ckpt.data-00000-of-00001':497759232,
            'model.ckpt.index':5215,
            'model.ckpt.meta':471155,
            'vocab.bpe':456318
        }

        for file_name, expected_size in files.items():
            url = f"{base_url}/{model_size}/{file_name}"
            valid,message = check_file_size(url, expected_size)
            assert valid, message
        
        model_size = '355M'
        files = {
            "checkpoint": 77,
            "encoder.json": 1042301,
            "hparams.json": 91,
            "model.ckpt.data-00000-of-00001": 1419292672,
            "model.ckpt.index": 10399,
            "model.ckpt.meta": 926519,
            "vocab.bpe": 456318
        }

        for file_name, expected_size in files.items():
            url = f"{base_url}/{model_size}/{file_name}"
            valid, message = check_file_size(url, expected_size)
            assert valid, message

    check_model_files(base_url="https://openaipublic.blob.core.windows.net/gpt-2/models")
    check_model_files(base_url="https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2")