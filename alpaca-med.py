#pip3 install gym transformers datasets openpyxl pandas sentencepiece protobuf protobuf==3.20 torch
import gym
import torch
import numpy as np
from torch.optim import Adam
from torch.distributions import Categorical
from typing import Tuple
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM, LlamaTokenizer
import pandas as pd


class PPO:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(
        self,
        model,
        tokenizer,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        num_epochs: int = 3,
        batch_size: int = 64,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

    def update(self, rewards, log_probs_old, states, actions):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for _ in range(self.num_epochs):
            log_probs_all, values = self.evaluate_actions(states, actions)
            for i in range(len(rewards)):
                advantages = rewards[i] - values[i].detach()
                ratio = (log_probs_all[i] - log_probs_old[i]).exp()

                obj = ratio * advantages
                obj_clipped = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                )
                policy_loss = -torch.min(obj, obj_clipped).mean()

                value_loss = (rewards[i] - values[i]).pow(2).mean()

                # Calculate the custom loss function
                custom_loss = policy_loss + value_loss

                # Enable gradient computation and calculate gradients
                #with torch.enable_grad():
                    #custom_loss.backward()

                # Update the model parameters without the optimizer

                for param in self.model.parameters():
                    if param.grad is not None:
                        param -= self.learning_rate * param.grad
                        # Manually zero gradients
                        param.grad.zero_()

    def act(self, state):
        with torch.no_grad():
            input_ids = self.tokenizer.encode(state, return_tensors="pt").to(self.device)
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]
            values = torch.tensor([0.0])  # We do not use values in this example
            action_probs = torch.softmax(logits, dim=-1)
            dist = Categorical(action_probs)
            action = dist.sample()

        return action.item(), dist.log_prob(action), values

    def evaluate_actions(self, states, actions) -> Tuple[torch.Tensor, torch.Tensor]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            input_ids = [self.tokenizer.encode(state) for state in states]
            input_ids = self.tokenizer.pad({"input_ids": input_ids}, return_tensors="pt").to(self.device)
            attention_mask = torch.ones_like(input_ids["input_ids"]).to(self.device)
            outputs = self.model(input_ids=input_ids["input_ids"], attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            values = torch.tensor([0.0] * len(states))  # We do not use values in this example
            action_probs = torch.softmax(logits, dim=-1)
            dist = Categorical(action_probs)
            log_probs = dist.log_prob(torch.tensor(actions).to(self.device))

        return log_probs, values

def create_dataset(model, tokenizer, prompts1, completions1, device) -> Tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # A very simple dataset to simulate human feedback
    prompts = prompts1
    completions = completions1
    rewards = [1.0, 1.0]

    states = []
    actions = []

    for prompt, completion in zip(prompts, completions):
        target_token_ids = tokenizer.encode(completion, add_special_tokens=False)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(PPO.device)
        attention_mask = torch.ones_like(input_ids).to(PPO.device)

        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
            generated_token_ids = outputs[0].tolist()

        # Only store actions corresponding to the target output
        actions.extend([generated_token_ids[i] for i in range(min(len(generated_token_ids), len(target_token_ids)))])
        for token_id in target_token_ids:
            states.append(prompt)
            prompt += f" {tokenizer.decode([token_id])}"

    return states, actions, rewards

def evaluate(model, tokenizer, input_sentences, expected_output_sentences, device):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct_count = 0
    total_count = len(input_sentences)

    for input_sentence, expected_output_sentence in zip(input_sentences, expected_output_sentences):
        prompt = input_sentence
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(PPO.device)

        with torch.no_grad():
            outputs = model.generate(input_ids)
            generated_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            print("The generated sentence is ", generated_sentence)

        if generated_sentence.strip() == expected_output_sentence:
            correct_count += 1

    return correct_count / total_count

def main():
    count = 0
    # Set the default device to GPU if available
    print(f"Using device: {PPO.device}")

    # Loading the RLHF dataset
    prompts =[]
    completions = []
    df = pd.read_excel('MedQuad dataset test.xlsx')
    print(df.columns)
    prompts =df['prompt']
    completions = df['completion']

    print(prompts)
    print(completions)
    print(type(prompts))
    print(type(completions))

    base_model = "huggyllama/llama-7b"
    model = LlamaForCausalLM.from_pretrained('huggyllama/llama-7b')
    model.to(PPO.device)  # Move the model to the specified device

    # Set requires_grad=True for all parameters in the model
    for param in model.parameters():
        param.requires_grad = True

    tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
    tokenizer.pad_token = tokenizer.eos_token

    states, actions, rewards = create_dataset(model, tokenizer, prompts, completions,PPO.device)
    ppo = PPO(model, tokenizer)

    # Evaluation data
    input_sentences = [
        "What is (are) Neck Injuries and Disorders ?",
        "What is (are) Heel Injuries and Disorders ?",
    ]
    expected_output_sentences = [
        "Any part of your neck  muscles, bones, joints, tendons, ligaments, or nerves  can cause neck problems. Neck pain is very common. Pain may also come from your shoulder, jaw, head, or upper arms. Muscle strain or tension often causes neck pain. The problem is usually overuse, such as from sitting at a computer for too long. Sometimes you can strain your neck muscles from sleeping in an awkward position or overdoing it during exercise. Falls or accidents, including car accidents, are another common cause of neck pain. Whiplash, a soft tissue injury to the neck, is also called neck sprain or strain. Treatment depends on the cause, but may include applying ice, taking pain relievers, getting physical therapy or wearing a cervical collar. You rarely need surgery.",
        "Heel problems are common and can be painful. Often, they result from too much stress on your heel bone and the tissues that surround it. That stress can come from  Injuries  Bruises that you get walking, running or jumping  Wearing shoes that don't fit or aren't made well  Being overweight These can lead to tendinitis, bursitis, and fasciitis, which are all types of inflammation of the tissues that surround your heel. Over time the stress can cause bone spurs and deformities. Certain diseases, such as rheumatoid arthritis and gout, can also lead to heel problems. Treatments for heel problems might include rest, medicines, exercises, taping, and special shoes. Surgery is rarely needed.",
    ]

    # Evaluate the base model
    base_model_score = evaluate(model, tokenizer, input_sentences, expected_output_sentences,PPO.device)
    print(f"Base model score: {base_model_score}")

    for _ in range(5):  # Train for 10 iterations
        count = count + 1
        print("The number of iteration is ", count)
        log_probs_old, values = [], []
        for state in states:
            action, log_prob, value = ppo.act(state)
            log_probs_old.append(log_prob.item())
            print("The log probabilities are", log_probs_old)
            values.append(value.item())
            print("The values are", values)

        log_probs_old = torch.tensor(log_probs_old)
        rewards = torch.tensor(rewards)
        print("The rewards are", rewards)
        ppo.update(rewards, log_probs_old, states, actions)
        print("PPO model updated number of iteration completed =", count)

    # Evaluate the PPO-trained model
    ppo_trained_model_score = evaluate(ppo.model, tokenizer, input_sentences, expected_output_sentences, PPO.device)
    print(f"PPO-trained model score: {ppo_trained_model_score}")

    print("Performance improvement:", ppo_trained_model_score - base_model_score)
    
    #Saving the Model on huggingface
    token = "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"
    model.push_to_hub("Amirkid/juicewrld-v1", use_auth_token=token)

if __name__ == "__main__":
    main()
