import torch
import torch.nn.functional as F
import copy


class PGDAttacker:
    def __init__(self, model, normalization, eps=8 / 255, steps=10, step_size=2 / 255):
        self.model = model
        self.normalize = normalization
        self.eps = eps
        self.steps = steps
        self.step_size = step_size

    def attack(self, images, device):
        images = images.detach().to(device)
        adv_images = images.clone() + torch.empty_like(images).uniform_(-self.eps, self.eps)
        adv_images = torch.clamp(adv_images, 0.0, 1.0)

        for _ in range(self.steps):
            adv_images.requires_grad_(True)
            normed = self.normalize(adv_images)
            output = self.model.inference_image(normed)
            features = output['image_feat']
            loss = -features.norm(dim=-1).mean()
            self.model.zero_grad()
            loss.backward()
            grad = adv_images.grad.data
            adv_images = adv_images + self.step_size * grad.sign()
            adv_images = torch.max(torch.min(adv_images, images + self.eps), images - self.eps)
            adv_images = torch.clamp(adv_images, 0.0, 1.0)

        return adv_images


class TextAttacker:
    def __init__(self, ref_model, tokenizer, max_length=30, number_perturbation=1, topk=10, threshold=0.3):
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_perturbation = number_perturbation
        self.topk = topk
        self.threshold = threshold
        self.device = ref_model.device

    def attack(self, net, texts, img_embeds=None):
        if img_embeds is None:
            raise ValueError("img_embeds must be provided for text attack.")
        
        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to(self.device)
        mlm_logits = self.ref_model(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        scores_all, predictions = torch.topk(mlm_logits, self.topk, dim=-1)

        with torch.no_grad():
            output = net.inference_text(text_inputs)
            original_embeds = output['text_feat']

        final_texts = []
        for i, text in enumerate(texts):
            words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            changed = 0

            importance = self._get_word_importance(text, net, original_embeds[i])
            ranked = sorted(enumerate(importance), key=lambda x: x[1], reverse=True)

            for idx, _ in ranked:
                if changed >= self.num_perturbation:
                    break
                token_range = keys[idx]
                if token_range[0] > self.max_length - 2:
                    continue
                subs = predictions[i, token_range[0]:token_range[1]]
                scores = scores_all[i, token_range[0]:token_range[1]]
                candidates = self._filter_substitutes(subs, scores, words[idx])
                if not candidates:
                    continue

                trial_texts = []
                word_choices = [words[idx]] + candidates
                for sub in word_choices:
                    temp = copy.deepcopy(final_words)
                    temp[idx] = sub
                    trial_texts.append(' '.join(temp))

                input_batch = self.tokenizer(trial_texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to(self.device)
                output_batch = net.inference_text(input_batch)['text_feat']
                sims = F.cosine_similarity(output_batch, img_embeds[i].unsqueeze(0).expand_as(output_batch))
                best = torch.argmin(sims)
                if word_choices[best] != words[idx]:
                    final_words[idx] = word_choices[best]
                    changed += 1

            final_texts.append(' '.join(final_words))

        return final_texts

    def _tokenize(self, text):
        words = text.split()
        keys = []
        idx = 0
        for word in words:
            sub_tokens = self.tokenizer.tokenize(word)
            keys.append([idx, idx + len(sub_tokens)])
            idx += len(sub_tokens)
        return words, keys

    def _filter_substitutes(self, ids, scores, original_word):
        result = []
        for tok_id, score in zip(ids, scores):
            if score < self.threshold:
                continue
            token = self.tokenizer._convert_id_to_token(int(tok_id))
            if '##' in token or token == original_word:
                continue
            result.append(token)
        return result

    def _get_word_importance(self, text, net, original_embed):
        masked_versions = []
        words = text.split()
        for i in range(len(words)):
            masked = words[:i] + ['[UNK]'] + words[i + 1:]
            masked_versions.append(' '.join(masked))
        
        inputs = self.tokenizer(masked_versions, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to(self.device)
        output = net.inference_text(inputs)['text_feat']
        loss = F.kl_div(output.log_softmax(dim=-1), original_embed.softmax(dim=-1).repeat(len(output), 1), reduction='none').sum(dim=-1)
        return loss.detach().cpu().tolist()


class Attacker:
    def __init__(self, model, img_attacker, txt_attacker):
        self.model = model
        self.img_attacker = img_attacker
        self.txt_attacker = txt_attacker

    def attack(self, imgs, txts, txt2img, device='cuda', **kwargs):
        adv_imgs = self.img_attacker.attack(imgs, device)

        with torch.no_grad():
            img_out = self.model.inference_image(adv_imgs)
            img_embeds = img_out['image_feat'][txt2img]

        adv_txts = self.txt_attacker.attack(self.model, txts, img_embeds=img_embeds)
        return adv_imgs, adv_txts, 0.0
