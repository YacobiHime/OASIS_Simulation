"""
Local LLM Adapters for OASIS Simulation
Supports Ollama and vLLM backends
"""

from typing import Optional, List, Dict, Any
import requests
import json
from abc import ABC, abstractmethod


class LocalLLMBase(ABC):
    """ローカルLLMの基底クラス"""
    
    def __init__(self, temperature: float = 0.7):
        self.temperature = temperature
        self._verify_connection()
    
    @abstractmethod
    def _verify_connection(self) -> bool:
        """接続確認"""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
    ) -> str:
        """テキスト生成"""
        pass
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: Optional[float] = None,
    ) -> str:
        """チャット形式での生成"""
        prompt = self._format_messages(messages)
        return self.generate(prompt, max_tokens, temperature)
    
    @staticmethod
    def _format_messages(messages: List[Dict[str, str]]) -> str:
        """メッセージをプロンプトに変換"""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        
        formatted.append("Assistant:")
        return "\n".join(formatted)


class OllamaLLM(LocalLLMBase):
    """Ollama統合"""
    
    def __init__(
        self,
        model: str = "mistral",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.generate_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"
        super().__init__(temperature)
    
    def _verify_connection(self) -> bool:
        """Ollamaサーバーへの接続確認"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            print(f"✓ Ollama接続成功: {self.base_url}")
            print(f"  使用モデル: {self.model}")
            return True
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Ollamaサーバーに接続できません。\n"
                f"  URL: {self.base_url}\n"
                f"  エラー: {e}\n\n"
                f"修正方法:\n"
                f"  1. Ollamaをインストール: curl https://ollama.ai/install.sh | sh\n"
                f"  2. サーバーを起動: ollama serve\n"
                f"  3. モデルをダウンロード: ollama pull {self.model}"
            )
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
    ) -> str:
        """生成エンドポイントを使用"""
        temp = temperature if temperature is not None else self.temperature
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": temp,
            "num_predict": max_tokens,
        }
        
        try:
            response = requests.post(self.generate_url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Ollamaレスポンスタイムアウト（120秒）\n"
                f"より小さいモデルを試してください: ollama pull neural-chat"
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama生成エラー: {e}")
    
    def generate_streaming(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
    ):
        """ストリーミング生成（ジェネレータ）"""
        temp = temperature if temperature is not None else self.temperature
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "temperature": temp,
            "num_predict": max_tokens,
        }
        
        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                stream=True,
                timeout=120
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield token
        except Exception as e:
            raise RuntimeError(f"Ollamaストリーミングエラー: {e}")


class VLLMLLMAdapter(LocalLLMBase):
    """vLLM統合（OpenAI互換API）"""
    
    def __init__(
        self,
        model: str = "mistralai/Mistral-7B-v0.1",
        base_url: str = "http://localhost:8000/v1",
        temperature: float = 0.7,
        api_key: str = "not-needed",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.client = None
        super().__init__(temperature)
    
    def _verify_connection(self) -> bool:
        """vLLMサーバーへの接続確認"""
        try:
            from openai import OpenAI
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            
            # テストリクエスト
            response = requests.get(f"{self.base_url}/models", timeout=5)
            response.raise_for_status()
            
            print(f"✓ vLLM接続成功: {self.base_url}")
            print(f"  使用モデル: {self.model}")
            return True
        except ImportError:
            raise ImportError(
                "openaiライブラリが必要です: pip install openai"
            )
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"vLLMサーバーに接続できません。\n"
                f"  URL: {self.base_url}\n"
                f"  エラー: {e}\n\n"
                f"修正方法:\n"
                f"  pip install vllm torch\n"
                f"  vllm serve {self.model} --host 0.0.0.0 --port 8000"
            )
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
    ) -> str:
        """生成"""
        if self.client is None:
            self._verify_connection()
        
        temp = temperature if temperature is not None else self.temperature
        
        try:
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                temperature=temp,
                max_tokens=max_tokens,
            )
            return response.choices[0].text.strip()
        except Exception as e:
            raise RuntimeError(f"vLLM生成エラー: {e}")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: Optional[float] = None,
    ) -> str:
        """チャット形式での生成"""
        if self.client is None:
            self._verify_connection()
        
        temp = temperature if temperature is not None else self.temperature
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"vLLMチャットエラー: {e}")


class LocalLLMFactory:
    """ローカルLLMファクトリー"""
    
    @staticmethod
    def create(
        backend: str = "ollama",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
    ) -> LocalLLMBase:
        """
        ローカルLLMインスタンスを生成
        
        Args:
            backend: "ollama" または "vllm"
            model: モデル名
            base_url: サーバーURL
            temperature: 温度パラメータ
        
        Returns:
            LocalLLMBase のサブクラスインスタンス
        """
        backend = backend.lower()
        
        if backend == "ollama":
            model = model or "mistral"
            base_url = base_url or "http://localhost:11434"
            return OllamaLLM(model=model, base_url=base_url, temperature=temperature)
        
        elif backend == "vllm":
            model = model or "mistralai/Mistral-7B-v0.1"
            base_url = base_url or "http://localhost:8000/v1"
            return VLLMLLMAdapter(model=model, base_url=base_url, temperature=temperature)
        
        else:
            raise ValueError(f"未知のバックエンド: {backend}")


# ============================================================================
# CAMEL統合用ラッパー
# ============================================================================

class CAMELLocalLLMAdapter:
    """CAMEL APIと互換性のあるアダプター"""
    
    def __init__(self, local_llm: LocalLLMBase):
        self.llm = local_llm
    
    def run(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """CAMEL互換のrun メソッド"""
        return self.llm.chat(messages, max_tokens=max_tokens, temperature=temperature)
    
    def generate_text(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """テキスト生成"""
        return self.llm.generate(prompt, **kwargs)


# ============================================================================
# 使用例
# ============================================================================

if __name__ == "__main__":
    print("🦙 ローカルLLMアダプターの動作確認\n")
    
    # 例1: Ollama
    try:
        print("=" * 60)
        print("例1: Ollamaを使用")
        print("=" * 60)
        
        ollama = OllamaLLM(model="mistral")
        
        response = ollama.generate(
            prompt="日本の首都は？",
            max_tokens=256
        )
        print(f"Q: 日本の首都は？\nA: {response}\n")
        
    except ConnectionError as e:
        print(f"❌ {e}\n")
    
    # 例2: vLLM
    try:
        print("=" * 60)
        print("例2: vLLMを使用")
        print("=" * 60)
        
        vllm = VLLMLLMAdapter(model="mistralai/Mistral-7B-v0.1")
        
        response = vllm.chat(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of Japan?"}
            ],
            max_tokens=256
        )
        print(f"Q: What is the capital of Japan?\nA: {response}\n")
        
    except ConnectionError as e:
        print(f"❌ {e}\n")
    
    # 例3: ファクトリーを使用
    print("=" * 60)
    print("例3: ファクトリーを使用")
    print("=" * 60)
    
    try:
        llm = LocalLLMFactory.create(backend="ollama", model="mistral")
        print("✓ ファクトリーで正常にLLMを作成しました\n")
    except Exception as e:
        print(f"ℹ️  {e}\n")
