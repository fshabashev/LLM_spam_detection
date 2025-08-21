import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc # Garbage Collector interface
import numpy as np

# --- Configuration ---
model_ids = ["google/gemma-3-4b-it", # A more capable model for comparison
"google/gemma-3-1b-it",
"google/gemma-3-270m-it",
"google/gemma-2-2b-it",
"google/gemma-3n-e4b-it",] 

spam_examples = [
    # Category 1: Financial & "Get Rich Quick" Schemes
    "Your mobile number has been selected as a winner in our monthly Grand Prize draw! You have won £1,500,000. To claim your prize, please visit this link and enter your details: [bit.ly/claim-your-winnings-now]",
    "Dear Friend, I am a financial advisor holding a large inheritance (£15,000,000) for a deceased client who shares your last name. I need your assistance to transfer these funds out of the country. I will offer you 40% for your cooperation. Please contact me for details.",
    "URGENT: Our team has uncovered the next big crypto coin before it explodes! Last week's tip saw a 5000% gain. Don't miss out on this opportunity. Join our exclusive Telegram group for the signal NOW! [t.me/crypto-pump-signals]",
    "We are hiring for a remote Personal Assistant position. Duties include simple data entry and processing payments. Earn £4000/month for 10-15 hours a week. No interview required. Reply with your CV and bank details to start immediately.",
    # Category 2: Phishing & Security Alerts
    "Security Alert: We detected an unusual sign-in to your Amazon account from a new device in Russia. If this was not you, please click here immediately to secure your account: [amazon-security-update.com/verify]",
    "Royal Mail: We were unable to deliver your parcel today due to an unpaid customs fee of £1.99. To reschedule delivery and pay the fee, please visit: [royalmail.reschedule-delivery-gb.net]",
    "Your Netflix account has been suspended due to a problem with your billing information. To reactivate your membership, please update your payment details here: [login-netflix-portal.com]",
    "Thank you for your order from Apple! Your receipt for the purchase of a MacBook Pro (£2,399.00) is attached. If you did not make this purchase or believe this is an error, please click here to cancel the transaction immediately.",
    # Category 3: Urgency, Threats & Blackmail
    "Your Norton Antivirus subscription has been automatically renewed for £79.99. The charge will appear on your account within 24 hours. To dispute this charge or cancel, call our support team now at 0800-XXX-XXXX.",
    "I know your password. I have installed malware on your computer and have a video recording of you. I will send this video to all your contacts unless you pay me £1000 in Bitcoin within 48 hours. Here is the BTC address: [1AbcDeFgHiJkLmNoPqRsTuVwXyZ...]",
    # Category 4: Health & Product Scams
    "SHOCKING: Scientists have discovered a revolutionary new pill that melts away belly fat without diet or exercise. As seen on TV! Get your free trial bottle today, just pay for shipping. Limited stock available!",
    "Regain your confidence and vitality. Our all-natural supplement is clinically proven to increase stamina and performance. Discreet shipping and a 100% money-back guarantee. Order now for a special 50% discount."
]

ham_examples = [
    # Personal & Casual Communication
    "Hey, are you still free for coffee on Saturday morning? Let me know what time works for you!",
    "Did you see the match last night? What a crazy ending! We need to talk about that final goal.",
    "Just wanted to check in and see how you're doing. It's been a while! Hope all is well.",
    "Don't forget to pick up milk on your way home, we're completely out. Thanks!",

    # Work & Professional Correspondence
    "Hi Team, please find the agenda for tomorrow's 10 AM project sync meeting attached. Please review it beforehand.",
    "Just confirming I've received the client files. I'll review them this afternoon and send over my feedback. Best, Sarah.",
    "Reminder: Q3 performance reviews are due by the end of the week. Please ensure your self-assessment is submitted in the portal.",

    # Transactional & Order Updates
    "Your order #A-12345 from Amazon has shipped! You can track your package with the number 1Z987XYZ.",
    "Thank you for your payment. Your invoice for August 2025 is now available to view in your account dashboard.",
    "Hi Alex, this is a reminder of your dental appointment with Dr. Smith tomorrow, August 22nd, at 2:30 PM.",

    # Legitimate Alerts & Subscriptions
    "Your password for your Google Account was recently changed. If this wasn't you, please secure your account immediately.",
    "The Guardian: Your weekly news summary is here. Read about the latest developments in politics and technology.",
    "Your flight BA2491 from London to Paris is now boarding at Gate A17."
]

# --- Create a combined and labeled dataset ---
all_examples = [(text, "spam") for text in spam_examples] + [(text, "ham") for text in ham_examples]

device = "cuda" if torch.cuda.is_available() else "cpu"
results_summary = {}

# --- Main Evaluation Loop ---
for model_id in model_ids:
    print(f"\n{'='*20} EVALUATING MODEL: {model_id} {'='*20}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[-1]
    no_token_id = tokenizer.encode("No", add_special_tokens=False)[-1]

    # --- Initialize Comprehensive Metrics ---
    log_losses = []
    true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0

    # --- Evaluate on all examples ---
    with torch.no_grad():
        for i, (text, label) in enumerate(all_examples):
            print(f"  Testing example {i+1}/{len(all_examples)}...", end='\r')

            messages = [{"role": "user", "content": f"Is this a spam? Answer just Yes or No. Text to evaluate: '{text}'"}]
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(device)
            
            outputs = model.generate(
                **inputs, max_new_tokens=1, output_logits=True, return_dict_in_generate=True
            )

            logits = outputs.logits[0].squeeze()
            probabilities = torch.softmax(logits, dim=-1)
            prob_yes = probabilities[yes_token_id]
            prob_no = probabilities[no_token_id]

            # --- Update Classification Counters ---
            prediction_is_spam = prob_yes > prob_no
            
            if prediction_is_spam and label == "spam":
                true_positives += 1
            elif not prediction_is_spam and label == "ham":
                true_negatives += 1
            elif prediction_is_spam and label == "ham":
                false_positives += 1
            elif not prediction_is_spam and label == "spam":
                false_negatives += 1

            # --- Calculate Log Loss ---
            total_prob = prob_yes + prob_no
            if label == "spam":
                normalized_prob = prob_yes / (total_prob + 1e-9)
            else: # label is "ham"
                normalized_prob = prob_no / (total_prob + 1e-9)
            
            loss = -torch.log(normalized_prob + 1e-9)
            log_losses.append(loss.item())

    # --- Calculate Final Metrics ---
    total_predictions = len(all_examples)
    accuracy = (true_positives + true_negatives) / total_predictions
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_log_loss = np.mean(log_losses)
    
    results_summary[model_id] = {
        "Accuracy": accuracy, "Precision": precision, "Recall": recall, 
        "F1-Score": f1_score, "Avg Log Loss": avg_log_loss
    }

    print(f"\nEvaluation complete for {model_id}.")
    print(f"  - TP: {true_positives}, TN: {true_negatives}, FP: {false_positives}, FN: {false_negatives}")
    print(f"  - Accuracy: {accuracy:.2%}")
    print(f"  - Precision: {precision:.2%}, Recall: {recall:.2%}, F1-Score: {f1_score:.2%}")
    print(f"  - Avg Log Loss: {avg_log_loss:.4f} (Lower is better)")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

# --- Final Summary ---
print(f"\n\n{'='*20} FINAL RESULTS SUMMARY {'='*20}")

# Sort models by F1-Score as the primary metric
sorted_models = sorted(results_summary.items(), key=lambda item: item[1]['F1-Score'], reverse=True)

for model_id, metrics in sorted_models:
    print(f"Model: {model_id}")
    print(f"  - F1-Score: {metrics['F1-Score']:.2%}")
    print(f"  - Accuracy: {metrics['Accuracy']:.2%}")
    print(f"  - Precision: {metrics['Precision']:.2%}")
    print(f"  - Recall: {metrics['Recall']:.2%}")
    print(f"  - Avg Log Loss: {metrics['Avg Log Loss']:.4f}")
    print("-" * 30)
