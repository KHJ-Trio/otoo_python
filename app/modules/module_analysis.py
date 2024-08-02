async def analyze_text(client, text, analysis_type):
    return await client.analyze(text, analysis_type)