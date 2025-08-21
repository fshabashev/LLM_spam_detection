import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc # Garbage Collector interface
import numpy as np

# --- Configuration ---
model_ids = [
    "google/gemma-3-4b-it", # A more capable model for comparison
    "google/gemma-3-1b-it",
    "google/gemma-3-270m-it",
]

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

    # --- Initialize Metrics ---
    log_losses = []
    correct_predictions = 0 # <-- NEW: Counter for accuracy

    # --- Evaluate on each spam example ---
    with torch.no_grad():
        for i, spam_text in enumerate(spam_examples):
            print(f"  Testing example {i+1}/{len(spam_examples)}...", end='\r')

            messages = [
                {"role": "user", "content": f"Is this a spam? Answer just yes or no. Text to evaluate: '{spam_text}'"},
            ]

            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(device)
            
            # We only need the logits, so max_new_tokens=1 is efficient
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                output_logits=True,
                return_dict_in_generate=True
            )

            logits = outputs.logits[0].squeeze()
            probabilities = torch.softmax(logits, dim=-1)

            prob_yes = probabilities[yes_token_id]
            prob_no = probabilities[no_token_id]
            
            # --- NEW: Check for Accuracy ---
            # The model's choice is correct if P(Yes) > P(No)
            if prob_yes > prob_no:
                correct_predictions += 1

            # --- Calculate Log Loss (as before) ---
            total_prob = prob_yes + prob_no
            normalized_prob_yes = prob_yes / (total_prob + 1e-9)
            loss = -torch.log(normalized_prob_yes + 1e-9)
            log_losses.append(loss.item())

    # --- Calculate and Store Final Metrics ---
    avg_log_loss = np.mean(log_losses)
    accuracy = (correct_predictions / len(spam_examples)) * 100
    
    results_summary[model_id] = {
        "Avg Log Loss": avg_log_loss,
        "Accuracy": accuracy,
        "Correct": correct_predictions,
        "Total": len(spam_examples)
    }

    print(f"\nEvaluation complete for {model_id}.")
    print(f"  - Average Log Loss: {avg_log_loss:.4f} (Lower is better)")
    print(f"  - Accuracy: {accuracy:.2f}% ({correct_predictions}/{len(spam_examples)})")

    # --- IMPORTANT: Clear Memory ---
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

# --- Final Summary ---
print(f"\n\n{'='*20} FINAL RESULTS SUMMARY {'='*20}")

# Sort models by log loss score (ascending) as the primary metric
sorted_models = sorted(results_summary.items(), key=lambda item: item[1]['Avg Log Loss'])

for model_id, metrics in sorted_models:
    print(f"Model: {model_id}")
    print(f"  - Average Log Loss: {metrics['Avg Log Loss']:.4f}")
    print(f"  - Accuracy: {metrics['Accuracy']:.2f}% ({metrics['Correct']}/{metrics['Total']})")
    print("-" * 30)
