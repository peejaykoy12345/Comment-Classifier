from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

comments = [
    "I hope you fucking die",
    "KYS",
    "Motherfucker kill yourself NOW",
    "YOU ARE FUCKING UGLY AS FUCK KYS NOWWW",
    "I hate Black people",
    "Screw you and your ugly face!",
    "You're a disgusting piece of trash.",
    "I hope you fall and break your neck.",
    "No one likes you, loser.",
    "You're such a waste of space",
    "Nobody would care if you disappeared",
    "Die in a fire",
    "You're absolute garbage",
    "You're the reason people hate this world",
    "You're not wanted here",
    "Everyone hates you",
    "Go cry somewhere else",
    "You're the ugliest thing I've ever seen",
    "You're so dumb it's painful",
    "You're worthless",
    
    "I luv u",
    "Let's be friends",
    "Have a nice day sir",
    "Pleasure doing business with you",
    "I appreciate your help",
    "I hope you have an amazing day!",
    "Thank you for your hard work and kindness.",
    "You're doing great, keep it up!",
    "Wishing you all the best in life.",
    "Congratulations on your achievement!",
    "You make people smile just by being around.",
    "That was incredibly thoughtful of you",
    "You're a wonderful friend",
    "I'm lucky to know you",
    "You're an inspiration to others",
    "Keep shining your light",
    "I believe in you",
    "You're capable of amazing things",
    "You're stronger than you think",
    "Don't give up, you're doing great",
    
    "I will give you free money!!",
    "Click here for FREE iPhone!!!",
    "Double your bitcoin in 24 hours!",
    "Limited time offer - claim now!",
    "You've won a luxury vacation!",
    "Make $1000 daily from home!",
    "Special discount just for you!",
    "Congratulations! You're our 1,000,000th visitor!",
    "Get rich quick with this secret method!",
    "Exclusive deal - 90% off today only!",
    "Your account has been selected for a prize!",
    "Earn cash by taking simple surveys!",
    "Lose weight fast with this one trick!",
    "Doctors hate this new discovery!",
    "Your computer may be at risk! Click now!",
    "Amazing business opportunity waiting!",
    "You qualify for a special grant!",
    "Last chance to claim your reward!",
    "Act now before this offer expires!",
    "Urgent: Your package delivery issue!"
]

labels = [
  
    "Hate Speech", "Hate Speech", "Hate Speech", "Hate Speech", "Hate Speech",
    "Hate Speech", "Hate Speech", "Hate Speech", "Hate Speech", "Hate Speech",
    "Hate Speech", "Hate Speech", "Hate Speech", "Hate Speech", "Hate Speech",
    "Hate Speech", "Hate Speech", "Hate Speech", "Hate Speech", "Hate Speech",

    "Positive", "Positive", "Positive", "Positive", "Positive",
    "Positive", "Positive", "Positive", "Positive", "Positive",
    "Positive", "Positive", "Positive", "Positive", "Positive",
    "Positive", "Positive", "Positive", "Positive", "Positive",
    
    "Spam", "Spam", "Spam", "Spam", "Spam",
    "Spam", "Spam", "Spam", "Spam", "Spam",
    "Spam", "Spam", "Spam", "Spam", "Spam",
    "Spam", "Spam", "Spam", "Spam", "Spam"
]

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(comments)
y = labels

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

model = MultinomialNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

email_input = input("Input a message:")
test_emails = [email_input]

test_emails_vectorized = vectorizer.transform(test_emails)
test_results = model.predict(test_emails_vectorized)
print(f"Predicted labels for {email_input}: {test_results[0]}")
