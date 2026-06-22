from app.schemas.chat import ProviderConfig


def create_model(config: ProviderConfig):
    if config.type == "openai_compatible":
        from langchain_openai import ChatOpenAI
        kwargs = {"api_key": config.api_key, "model": config.model}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)

    if config.type == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(api_key=config.api_key, model=config.model)

    if config.type == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=config.model, google_api_key=config.api_key)

    raise ValueError(f"Unsupported provider type: {config.type}")
