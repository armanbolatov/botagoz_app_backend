# Botagoz App Backend

This is the backend for the Botagoz App. It is an Android application for AI assistant designed specifically to empower individuals with visual impairments, which provides users with a seamless and supportive experience in navigating their surroundings. See the [frontend repository](https://github.com/ulpanb123/Botagoz-App) for more information.

## Algorithm

<img src="images/algorithm.png" width="750">

An Android phone sends the photo and the question to the FastAPI backend. It appends the question to the chat log. Then, a classifier based on ChatGPT analyzes whether it needs visual context. If so, it uses BLIP via the Replicate.com API and processes it. If not, it answers the question as it is. 