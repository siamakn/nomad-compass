from nomad.config.models.plugins import APIEntryPoint


class ChatbotAPIEntryPoint(APIEntryPoint):

    def load(self):
        from nomad_compass.apis.chatbot_api import app

        return app


chatbot_api = ChatbotAPIEntryPoint(
    prefix = 'chatbot-api',
    name = 'ChatBot API',
    description = 'chatbot api description.',
)