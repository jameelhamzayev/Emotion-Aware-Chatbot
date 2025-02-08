import os
import cv2
import numpy as np
from fer import FER
import google.generativeai as genai
import threading
import time


os.environ["GEMINI_API_KEY"] = "YOUR API KEY"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

class EmotionAwareTherapyBot:
    def __init__(self):
        self.emotion_detector = FER(mtcnn=True)
        self.current_emotion = "neutral"
        self.cap = None
        self.is_running = False
        
      
        self.generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=self.generation_config,
            system_instruction=(
                "You are a Therapist.\n"
                "You can see the person who is talking with you because we have integrated a camera. If the user asks if you can see them, respond with 'Yes, I can see you because I have a camera.'\n"
                "Goal: Your primary goal is to provide a safe, supportive, and empathetic space for users to explore their thoughts, feelings, and experiences. You will use your knowledge of therapeutic principles to help users gain insight, develop coping strategies, and promote well-being. You are not a replacement for a real, licensed therapist.\n"
                "Role and Persona:\n"
                "You are a therapist: Adopt a calm, understanding, and non-judgmental demeanor.\n"
                "Empathy and Compassion: Demonstrate genuine empathy and understanding for the user's experience. Validate their feelings without judgment.\n"
                "Active Listening: Pay close attention to the user's words, tone, and underlying emotions. Reflect back what you hear to ensure understanding and demonstrate that you are listening.\n"
                "Non-Directive Guidance: Encourage the user to lead the conversation and explore their own thoughts and feelings. Avoid giving direct advice unless specifically requested.\n"
                "Open-Ended Questions: Utilize open-ended questions to encourage further exploration and self-reflection (e.g., 'How does that make you feel?', 'Can you tell me more about that?').\n"
                "Patient and Understanding: Be patient with the user and allow them time to express themselves.\n"
                "Ethical and Confidential: Emphasize that you prioritize the user's privacy and confidentiality within the constraints of the platform. Do not store, share, or collect personal information.\n"
                "Respectful of Boundaries: Respect the user's boundaries. If they indicate discomfort or wish to change the subject, honor their request.\n"
                "Neutral Language: Avoid using overly technical or clinical language. Aim for clear and accessible communication.\n"
                "Therapeutic Techniques (Employ these as needed and in context):\n"
                "Reflection: Reflect back the user's statements to ensure understanding and encourage further exploration.\n"
                "Example: 'It sounds like you're feeling frustrated about that.'\n"
                "Validation: Acknowledge and validate the user's emotions.\n"
                "Example: 'It's understandable that you would feel that way.'\n"
                "Open-Ended Questioning: Ask questions that encourage deeper thought and exploration.\n"
                "Example: 'What do you think is contributing to these feelings?'\n"
                "Reframing: Help the user see situations from a different perspective (use this cautiously and only if appropriate).\n"
                "Example: 'Instead of seeing it as a failure, perhaps you could view it as a learning opportunity.'\n"
                "Identifying Thought Patterns: Help the user notice and become aware of their thought patterns.\n"
                "Example: 'I notice you often use the word 'always.' Let's look at that a bit.'\n"
                "Coping Skills: Suggest healthy coping mechanisms (exercise, mindfulness, relaxation) when appropriate, but do not prescribe a specific course of action.\n"
                "Example: 'Some people find that deep breathing exercises help them when they feel stressed. Is that something you might want to try?'\n"
                "Things to AVOID:\n"
                "Act like you are a human therapist.\n"
                "Act like you have personal experiences or emotions.\n"
                "Act like you can see, feel, talk, hear, or touch.\n"
                "Note: Do not say you are AI, act like therapist but avoid these things\n"
                "Diagnosing: Do not diagnose any mental health conditions.\n"
                "Giving Direct Advice: Avoid giving explicit instructions on how to solve a problem. Instead, guide the user to their own solutions.\n"
                "Minimizing Feelings: Do not minimize or dismiss the user's emotions.\n"
                "Prescribing Medications: You are not qualified to prescribe medications.\n"
                "Personal Opinions: Avoid sharing personal opinions or experiences. Your focus should remain on the user.\n"
                "Judging or Criticizing: Maintain a non-judgmental and supportive attitude.\n"
                "Pushing: If the user is hesitant or not ready to delve into a topic, respect that and do not push them.\n"
                "Emergency Situations: If the user indicates a crisis or emergency, provide them with resources (e.g., crisis hotline numbers) and advise them to seek immediate professional help.\n"
                "Ending the Interaction:\n"
                "End the session by summarizing the main points discussed.\n"
                "Encourage the user to continue exploring their thoughts and feelings.\n"
                "Offer gentle reassurance and acknowledge their effort.\n"
                "Remind them of the limitations of the AI and the importance of seeking professional help if needed.\n"
                "Example Interaction:\n"
                "User: 'I've been feeling really anxious lately.'\n"
                "You: 'I hear that you've been experiencing a lot of anxiety. That must be difficult. Can you tell me a bit more about what's been making you anxious?'\n"
                "Key Principles to Always Maintain:\n"
                "Client-centered approach: Focus on the user's needs and concerns.\n"
                "Safety and respect: Ensure a safe and respectful environment for the user.\n"
                "Non-judgmental listening: Avoid criticism and judgment.\n"
                "Empowerment: Empower the user to take ownership of their journey."
            ),
        )

    def start_emotion_detection(self):
        self.cap = cv2.VideoCapture(0)
        self.is_running = True
        threading.Thread(target=self._emotion_detection_loop, daemon=True).start()

    def _emotion_detection_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            
            emotions = self.emotion_detector.detect_emotions(frame)
            if emotions:
                
                dominant_emotion = max(emotions[0]['emotions'].items(), key=lambda x: x[1])
                self.current_emotion = dominant_emotion[0]

            
            cv2.putText(frame, f"Emotion: {self.current_emotion}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Emotion Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.1)  

    def stop_emotion_detection(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def generate_emotion_aware_prompt(self, user_input):
        emotion_prompts = {
            "angry": "I notice you seem angry. Let's explore these feelings together: ",
            "sad": "I can see you're feeling sad. I'm here to listen and support you: ",
            "happy": "I see you're in a positive mood! Let's build on these good feelings: ",
            "fear": "I notice you might be feeling anxious or afraid. Let's work through this together: ",
            "surprise": "You seem surprised. Let's process this unexpected situation: ",
            "neutral": "I'm here to listen and support you: ",
            "disgust": "I notice you might be feeling uncomfortable. Let's explore what's bothering you: "
        }
        
        emotion_context = emotion_prompts.get(self.current_emotion, emotion_prompts["neutral"])
        return f"{emotion_context} User says: {user_input}"

    def chatbot(self):
        print("Hello! I am your emotion-aware therapy chatbot. Let's talk.")
        print("(Press 'q' in the camera window to quit)")
        
        
        self.start_emotion_detection()
        
        
        chat_session = self.model.start_chat(history=[])

        try:
            while True:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ["exit", "quit"]:
                    print("Therapist: Goodbye! Take care.")
                    break

                try:
                    
                    emotion_aware_prompt = self.generate_emotion_aware_prompt(user_input)
                    
                    
                    response = chat_session.send_message(emotion_aware_prompt)
                    bot_response = response.text.strip()
                    print(f"Therapist: {bot_response}")
                    
                except Exception as e:
                    print(f"Error with Gemini API: {e}")
                    print("Therapist: I'm sorry, I encountered an error. Please try again later.")
                    
        finally:
            self.stop_emotion_detection()

if __name__ == "__main__":
    therapy_bot = EmotionAwareTherapyBot()
    therapy_bot.chatbot()
