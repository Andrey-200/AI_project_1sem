import ollama

response = ollama.chat(
    model='qwen3:8b',
    messages=[
        {
            'role': 'system',
            'content': 'Ты помощник для анализа российских судебных документов.'
        },
        {
            'role': 'user',
            'content': 'Кратко объясни, чем занимается Legal Norm Auditor.'
        }
    ]
)

print(response['message']['content'])