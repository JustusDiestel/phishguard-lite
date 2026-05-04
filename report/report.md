# PhishGuard-Lite: A Lightweight Hybrid Phishing Detection System Using Feature Selection and Pretrained Hugging Face Models

**Topic:** Lightweight Machine Learning-Based Phishing Detection with Feature Selection and Pretrained Models: A Practical Evaluation Study

**Project type:** University machine learning and cybersecurity prototype

---

## 1. Introduction

Phishing remains one of the most common cybersecurity threats faced by individuals, universities, companies, and public institutions. In a phishing attack, an attacker attempts to deceive a user into visiting a fraudulent website, opening a malicious link, or submitting sensitive information such as usernames, passwords, payment details, or account recovery codes. Although many phishing attacks are distributed through email, text messages, social media, and instant messaging platforms, the final point of interaction is often a URL. For this reason, phishing URL detection is an important defensive task.

Traditional phishing protection often depends on blacklists, browser warnings, or manually maintained threat feeds. These mechanisms are useful, but they have an important limitation: a newly created phishing URL may not appear in a blacklist until after users have already received it. Attackers also frequently modify domains, subdomains, paths, and query strings to avoid exact-match detection. A lightweight machine learning approach can help address this problem by learning suspicious patterns from URL structure rather than relying only on known malicious addresses.

This project presents **PhishGuard-Lite**, a practical phishing URL detection prototype. The system combines handcrafted URL features, feature selection, classical machine learning models, optional pretrained Hugging Face models, and a transparent hybrid risk score. The goal is not to claim production-level phishing protection, but to build a clean and explainable student project that demonstrates how multiple detection strategies can be combined in a lightweight way.

The project supports two main use cases. First, a user can enter a single URL and receive a prediction, risk level, confidence score, model comparison, suspicious indicators, and extracted feature table. Second, a labeled CSV dataset can be evaluated to produce common classification metrics such as accuracy, precision, recall, F1-score, true positive rate, false positive rate, confusion matrix, feature importance, and latency estimates.

The project is intentionally designed to be runnable on a local machine using Python, scikit-learn, Streamlit, and optional Hugging Face Transformers. If the Hugging Face layer cannot be loaded because of missing internet access or limited hardware resources, the system continues to operate using the classical machine learning model and rule-based scoring layer.

## 2. Background

Phishing detection can be approached in several ways. Blacklist-based systems compare a URL against a database of known malicious links. Content-based systems inspect the HTML, text, forms, scripts, and visual structure of a webpage. Network-based systems may use DNS records, WHOIS information, TLS certificate data, hosting information, or domain age. Machine learning systems attempt to learn patterns that separate phishing and legitimate examples.

This project focuses on **URL-based phishing detection**. URL-based detection is attractive for a lightweight prototype because the system does not need to crawl websites, execute scripts, or collect webpage content. A URL can be analyzed quickly and safely as text. Useful signals include URL length, number of subdomains, number of hyphens, use of IP addresses, presence of suspicious keywords, use of URL shorteners, suspicious top-level domains, and unusually high character entropy.

However, URL-based detection also has limitations. A legitimate URL may contain words such as "login" or "account" because many real websites have account pages. Similarly, a phishing URL may use a short and clean-looking domain. Therefore, individual URL features should not be treated as proof of phishing. Instead, they are best used as evidence that contributes to a risk estimate.

Classical machine learning models are suitable for this type of structured feature data. Logistic Regression provides a simple linear baseline. Random Forest can learn nonlinear patterns and provides feature importance. Linear Support Vector Machines can work well on small numeric feature sets. Gradient Boosting can also capture nonlinear relationships. In this project, the Random Forest model is used as the main model because it is lightweight, works well with tabular features, and supports feature importance analysis.

Pretrained transformer models offer a different perspective. Instead of relying only on manually selected numeric features, a pretrained URL model can process the URL as text and learn token patterns from larger training data. Hugging Face provides access to pretrained models for phishing or malicious URL classification. These models can improve generalization in some cases, but they may require internet access, disk space, and more computing resources. Therefore, this project treats the Hugging Face layer as optional and includes fallback logic.

## 3. Related Work and Paper Connection

The referenced academic topic emphasizes lightweight machine learning-based phishing detection, feature selection, and pretrained or deep learning models. PhishGuard-Lite is connected to that direction in four main ways. First, it uses phishing URL detection as the target cybersecurity problem. Second, it extracts structured URL features and evaluates classical machine learning models. Third, it applies feature selection to compare smaller feature subsets against the full feature set. Fourth, it integrates optional pretrained Hugging Face models as a deep learning component.

It is important to state clearly that this project is a **practical adaptation**, not a full reproduction of the referenced paper. A full reproduction would require the same dataset, the same preprocessing assumptions, the same experimental protocol, the same model configurations, and direct comparison with the original paper's results. This project does not attempt that. Instead, it implements the central ideas in a smaller, classroom-suitable system.

The practical adaptation is useful because it turns the paper's general research direction into a working prototype. The system allows students to test URLs, inspect features, evaluate models, view confusion matrices, and discuss the trade-offs between handcrafted features, feature selection, classical machine learning, pretrained models, and explainability. The project also demonstrates honest limitations: performance on a small sample dataset should not be interpreted as performance on real-world phishing traffic.

##  3.1 Why are Phishing URLS detectable?
Phishing-URLs lassen sich erkennen, weil Angreifer bei der Erstellung solcher Links grundlegenden Einschränkungen unterliegen. Insbesondere können sie keine legitimen Domains wie etwa die von Banken oder großen Plattformen frei verwenden, da diese bereits registriert und geschützt sind. Stattdessen müssen sie auf alternative Domainnamen ausweichen, die der ursprünglichen Adresse lediglich ähneln. Dadurch entstehen zwangsläufig Abweichungen, die sich analysieren und als Muster identifizieren lassen.

Um dennoch Vertrauen zu erwecken, enthalten Phishing-URLs häufig bestimmte Schlüsselbegriffe wie „login“, „secure“ oder „verify“. Diese sollen Seriosität vortäuschen, führen jedoch gleichzeitig zu charakteristischen Strukturen, die von Erkennungssystemen genutzt werden können. Zusätzlich greifen Angreifer oft auf neu registrierte oder kostengünstige Domains zurück, da Phishing-Seiten in der Regel schnell entdeckt und wieder entfernt werden. Auch dies ist ein unterscheidbares Merkmal, da legitime Webseiten meist eine längere Historie besitzen.

Ein weiterer wichtiger Punkt ist, dass Phishing-Angriffe primär darauf abzielen, menschliche Nutzer zu täuschen und nicht unbedingt automatisierte Systeme perfekt zu umgehen. Viele Nutzer prüfen URLs nur oberflächlich, wodurch bereits leicht veränderte oder verlängerte Adressen überzeugend wirken können. Technisch betrachtet enthalten diese jedoch oft Auffälligkeiten wie ungewöhnlich lange URLs, viele Subdomains, Sonderzeichen oder irreführende Domainstrukturen. Ein Beispiel dafür ist eine URL, bei der der eigentliche Domainname am Ende steht, während bekannte Begriffe wie „paypal“ nur als Teil einer Subdomain erscheinen.

Die Erkennung von Phishing basiert daher nicht auf einer einzelnen Eigenschaft, sondern auf der Kombination mehrerer Merkmale. Maschinelle Lernverfahren bewerten diese Merkmale und bestimmen auf Basis statistischer Muster, wie wahrscheinlich es ist, dass es sich um eine betrügerische URL handelt. Dabei handelt es sich nicht um eine absolute Entscheidung, sondern um eine Wahrscheinlichkeitsaussage. Besonders gut gemachte Phishing-Seiten können daher schwer zu erkennen sein, weshalb zusätzlich weitere Informationen wie Webseiteninhalt, Zertifikate oder Domain-Metadaten einbezogen werden.

Zusammenfassend lässt sich sagen, dass Phishing-URLs erkennbar sind, weil Angreifer gezwungen sind, Kompromisse einzugehen. Sie können die Identität legitimer Anbieter nicht vollständig imitieren und greifen daher auf Strategien zurück, die zwar für Menschen oft überzeugend wirken, aber gleichzeitig messbare und wiederkehrende Muster erzeugen. Diese Muster bilden die Grundlage für moderne Erkennungssysteme.

Echte: https://www.paypal.com/login

Fake:   https://paypal-login-security.com
        https://paypaI.com/login
        https://paypal.com.verify-account.ru
        https://bit.ly/abc123

## 3.2 Attributes to detect phishing URLs
Das Paper zeigt, dass bei der Erkennung von Phishing-Webseiten nicht primär die Anzahl der verwendeten Attribute entscheidend ist, sondern deren Aussagekraft. Zwar wurden zunächst sehr viele Merkmale aus den URLs und Webseiten extrahiert, jedoch führte nicht die bloße Menge dieser Merkmale zum besten Ergebnis. Stattdessen zeigte sich, dass eine gezielte Auswahl weniger, aber relevanter Attribute nahezu dieselbe oder sogar eine bessere Erkennungsleistung ermöglichen kann. Das ist ein wichtiger Punkt, weil viele Attribute auch unnötige Informationen oder Rauschen enthalten können. Solche irrelevanten Merkmale verbessern das Modell nicht, sondern können es sogar schwächen, da es Zusammenhänge lernt, die für die eigentliche Unterscheidung zwischen legitimen und betrügerischen Webseiten keine Bedeutung haben.
Die zentrale Aussage ist daher, dass Qualität wichtiger ist als Quantität. Ein kleines Set gut ausgewählter Attribute kann typische Phishing-Muster bereits ausreichend abbilden, beispielsweise auffällige URL-Strukturen, verdächtige Schlüsselbegriffe, ungewöhnliche Domain-Eigenschaften oder technische Merkmale der Webseite. Wenn diese Merkmale stark mit Phishing zusammenhängen, liefern sie dem Modell genügend Information, um zuverlässige Entscheidungen zu treffen. Zusätzliche Attribute bringen dann nur noch geringen Nutzen, erhöhen aber den Rechenaufwand und können das Risiko von Overfitting steigern.
Gerade für praktische Anwendungen ist diese Erkenntnis relevant. Ein Modell mit weniger Attributen ist schneller, ressourcenschonender und leichter einsetzbar, etwa in Echtzeitsystemen oder Browser-Schutzmechanismen. Das Paper macht damit deutlich, dass ein gutes Phishing-Erkennungssystem nicht möglichst viele Daten sammeln muss, sondern die richtigen Merkmale identifizieren sollte. Die Leistung eines Modells hängt also weniger davon ab, wie umfangreich der Merkmalskatalog ist, sondern davon, ob die ausgewählten Attribute tatsächlich zwischen legitimen und betrügerischen Webseiten unterscheiden können.

## 3.3 Quanta Computers Phishing Case
Ein bekanntes Beispiel für erfolgreiches Phishing im großen Unternehmenskontext ist der sogenannte „Quanta Computer“-Betrug, bei dem unter anderem Google und Facebook betroffen waren. In diesem Fall nutzte der Angreifer keine offensichtlich auffälligen oder technisch komplexen Methoden, sondern setzte gezielt auf täuschend echt wirkende Domainnamen und glaubwürdige Kommunikationsinhalte. Er registrierte Domains, die dem tatsächlichen Zulieferer Quanta Computer ähnelten, beispielsweise Varianten wie „quanta-computer.com“ oder „quantacomputers.com“. Diese unterscheiden sich nur minimal von legitimen Adressen und sind für menschliche Empfänger auf den ersten Blick kaum als betrügerisch erkennbar.

Auf Basis dieser Domains wurden anschließend E-Mails versendet, die wie legitime Geschäftskommunikation wirkten, insbesondere in Form von Rechnungen oder Zahlungsaufforderungen. Da große Unternehmen regelmäßig mit zahlreichen Lieferanten interagieren, erschien diese Kommunikation plausibel und wurde nicht ausreichend hinterfragt. Entscheidend ist hierbei, dass es sich nicht um klassisches Phishing mit gefälschten Login-Seiten handelte, sondern um eine Form des sogenannten Business Email Compromise. Dabei wird weniger auf technische Täuschung als vielmehr auf Vertrauen und organisatorische Abläufe gesetzt.

Der Angriff war insofern erfolgreich, als dass Mitarbeiter die Anweisungen aus den E-Mails befolgten und Zahlungen an die vom Angreifer kontrollierten Konten überwiesen. Insgesamt entstand ein Schaden von über 100 Millionen US-Dollar. Dieses Beispiel verdeutlicht, dass Phishing nicht zwangsläufig durch auffällige oder fehlerhafte URLs gekennzeichnet ist. Vielmehr können bereits geringfügige Abweichungen in Domainnamen ausreichen, um glaubwürdig zu erscheinen, insbesondere wenn sie in einen realistischen Kontext eingebettet sind.

## 4. System Design

PhishGuard-Lite is organized as a modular Python project. The main components are:

- Feature extraction module
- Feature selection module
- Classical machine learning training module
- Hugging Face model wrapper
- Hybrid detector
- Explanation module
- Evaluation and metrics module
- Streamlit interface
- Report asset generation script

The system accepts either a single URL or a labeled CSV file. For single URL analysis, the system extracts handcrafted features, obtains a classical machine learning phishing score, optionally obtains a Hugging Face score, computes a rule-based risk score, combines the scores into a final hybrid score, and returns an explanation. For batch evaluation, the system predicts labels for a dataset and computes evaluation metrics.

**Figure 1. System architecture placeholder**

![Figure 1: System architecture placeholder](figures/system_architecture_placeholder.png)

The modular design supports classroom demonstration because each part can be explained separately. For example, a teacher can ask how the URL is converted into features, how feature selection works, how the Random Forest produces a score, how fallback behavior works, or how the final hybrid score is calculated.

## 5. Feature Engineering

Feature engineering converts each URL into a numeric vector. The extractor is designed to be robust against malformed input and missing schemes. If a URL does not include `http://` or `https://`, the parser adds a temporary scheme so that the URL can still be analyzed.

The project extracts structural, lexical, and keyword-based features. Structural features describe the length and shape of the URL, such as total URL length, domain length, path length, query length, number of dots, number of slashes, and path depth. Lexical features include counts of hyphens, digits, special characters, and entropy. Binary indicator features identify specific suspicious patterns such as an IP address, `@` symbol, double-slash redirect pattern, suspicious top-level domain, and URL shortener. Keyword indicators detect words commonly seen in phishing lures, such as `login`, `verify`, `secure`, `account`, `update`, and `bank`.

**Table 1. Extracted URL features**

| Feature | Description |
|---|---|
| `url_length` | Total number of characters in the URL |
| `domain_length` | Number of characters in the host/domain |
| `path_length` | Number of characters in the URL path |
| `query_length` | Number of characters in the query string |
| `num_dots` | Count of dot characters |
| `num_hyphens` | Count of hyphen characters |
| `num_digits` | Count of numeric characters |
| `num_special_chars` | Count of non-alphanumeric characters |
| `num_slashes` | Count of slash characters |
| `subdomain_count` | Number of subdomain parts |
| `path_depth` | Number of path segments |
| `has_https` | Whether the URL uses HTTPS |
| `has_ip_address` | Whether the host appears to be an IPv4 address |
| `has_at_symbol` | Whether the URL contains an `@` symbol |
| `has_double_slash_redirect` | Whether path or query contains a double-slash redirect-like pattern |
| `has_suspicious_tld` | Whether the top-level domain appears in a suspicious TLD list |
| `has_url_shortener` | Whether the registered domain is a known URL shortener |
| `contains_*` | Keyword indicators for login, verify, secure, account, update, bank, and brands |
| `entropy_score` | Shannon entropy of the URL string |

These features are not perfect indicators. For example, HTTPS does not guarantee legitimacy because phishing sites can also use TLS certificates. Likewise, a legitimate banking site may contain the word "login". The purpose of feature engineering is to provide a set of signals that a model can weigh together.

## 6. Feature Selection

Feature selection is included for two reasons. First, it supports the academic focus on lightweight phishing detection. A smaller feature set can reduce complexity, improve interpretability, and lower computation cost. Second, it provides a way to study which URL features are most useful for the model.

The project implements two feature selection approaches:

- **Permutation importance:** measures how model performance changes when each feature is randomly shuffled.
- **SelectKBest with mutual information:** estimates statistical dependence between each feature and the phishing label.

The evaluation compares the full feature set against the top 20, top 10, and top 5 features selected by mutual information. Feature lists are saved as JSON files so that they can be reused or inspected. The report asset script generates a feature selection comparison chart.

**Figure 5. Feature selection comparison placeholder**

![Figure 5: Feature selection comparison](figures/feature_selection_comparison.png)

**Table 3. Feature selection results**

| Feature set | Number of features | F1-score |
|---|---:|---:|
| All features | 29 | 0.8571 |
| Top 20 mutual information features | 20 | 0.8571 |
| Top 10 mutual information features | 10 | 0.8571 |
| Top 5 mutual information features | 5 | 0.9091 |

## 7. Model Design

The project trains several classical machine learning models:

- Logistic Regression
- Random Forest
- Linear SVM
- Gradient Boosting

The Random Forest is selected as the main model because it performs well on structured tabular features, can model nonlinear interactions, and provides feature importance values. Logistic Regression is useful as a simple baseline. Linear SVM provides another lightweight classifier, although its confidence score requires conversion from the decision function. Gradient Boosting is included as an optional comparison model.

The training process reads a CSV file, extracts URL features, splits the data into training and test sets, trains each model, evaluates classification metrics, and saves the trained model bundles using Joblib. The main saved model is `models/phishguard_rf.joblib`.

**Table 2. Model comparison**

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9167 | 1.0000 | 0.8333 | 0.9091 |
| Random Forest | 0.8750 | 1.0000 | 0.7500 | 0.8571 |
| Linear SVM | 0.9167 | 1.0000 | 0.8333 | 0.9091 |
| Gradient Boosting | 0.9167 | 1.0000 | 0.8333 | 0.9091 |

## 8. Hugging Face Pretrained Model Integration

The Hugging Face layer is implemented as a lazy-loading wrapper around pretrained URL classification models. The wrapper can attempt to load models such as:

- `CrabInHoney/urlbert-tiny-v4-phishing-classifier`
- `CrabInHoney/urlbert-tiny-v4-malicious-url-classifier`
- `Eason918/malicious-url-detector-v2`
- `darshan8950/phishing_url_detection_BERT`

The model is loaded only when the user enables the Hugging Face layer and requests a prediction. This design avoids unnecessary startup cost and makes offline classroom demonstrations more reliable. If the model cannot be loaded because of missing dependencies, missing internet access, model download failure, or hardware constraints, the wrapper returns an unavailable result instead of crashing the application.

This layer represents the pretrained/deep learning component of the project. It is not treated as mandatory because pretrained transformer models are heavier than the classical feature-based model. Instead, it provides an optional comparison signal for the hybrid risk engine.

## 9. Hybrid Risk Scoring and Explanation Layer

The hybrid risk engine combines three sources of evidence:

- Classical machine learning phishing score
- Hugging Face phishing score, when available
- Rule-based risk score

When the Hugging Face layer is available, the formula is:

```text
final_score = 0.40 * classical_ml_score + 0.40 * hf_score + 0.20 * rule_score
```

When the Hugging Face layer is disabled or unavailable, the fallback formula is:

```text
final_score = 0.70 * classical_ml_score + 0.30 * rule_score
```

The final score is interpreted using the following thresholds:

- LOW risk: final score below 0.40
- MEDIUM risk: final score from 0.40 to below 0.70
- HIGH risk: final score of 0.70 or higher

The final prediction is phishing when the final score is at least 0.50. Otherwise, the prediction is legitimate.

The explanation layer produces simple human-readable reasons based on activated URL features. Example explanations include: "URL is unusually long", "URL contains an IP address instead of a normal domain", "URL contains credential-related keyword: login", and "URL contains a brand name outside the official domain." These explanations are not full explainable AI in the formal sense. They are rule-based summaries intended to help a non-expert understand why the URL appears suspicious.

## 10. Experimental Setup

The prototype includes a small safe demonstration dataset located at `data/sample_urls.csv`. The dataset has two columns: `url` and `label`, where `0` means legitimate and `1` means phishing. The phishing examples are synthetic or reserved-domain style examples and are included only for educational testing. They are not real credential-stealing websites.

The experimental pipeline consists of the following steps:

1. Load the labeled URL dataset.
2. Extract handcrafted URL features.
3. Split the dataset into training and test sets.
4. Train classical machine learning models.
5. Compare model metrics.
6. Run feature selection using mutual information and permutation importance.
7. Evaluate the saved Random Forest model.
8. Generate confusion matrix, feature importance, feature selection, model comparison, and latency assets.

The main evaluation metrics are accuracy, precision, recall, F1-score, true positive rate, false positive rate, confusion matrix, and average prediction latency. In phishing detection, recall is especially important because false negatives mean phishing URLs are missed. However, precision is also important because too many false positives can reduce user trust.

**Figure 2. Streamlit interface placeholder**

![Figure 2: Streamlit interface placeholder](figures/streamlit_interface_placeholder.png)

## 11. Results

The exact results depend on the dataset, train/test split, installed dependencies, and whether the optional Hugging Face model is enabled. The included sample dataset is small and should be interpreted as a demonstration dataset, not a real benchmark.

After running:

```bash
python train.py
python generate_report_assets.py
```

the project generates tables in `results/tables/` and figures in `results/figures/` and `report/figures/`.

On the included demonstration dataset, the saved Random Forest model achieved an overall evaluation accuracy of 0.9625, precision of 1.0000, recall of 0.9250, and F1-score of 0.9610 when evaluated across the sample CSV. The false positive rate was 0.0000 and the true positive rate was 0.9250. These values are useful for checking that the implementation works, but they should not be interpreted as real-world benchmark results because the dataset is small and contains synthetic phishing-style examples.

**Figure 3. Confusion matrix placeholder**

![Figure 3: Confusion matrix](figures/confusion_matrix.png)

**Figure 4. Feature importance placeholder**

![Figure 4: Feature importance](figures/feature_importance.png)

**Table 4. Latency comparison**

| Component | Average latency | Notes |
|---|---:|---|
| Classical ML model | 0.003701 seconds | Random Forest over handcrafted features |
| Rule-based layer | 0.000027 seconds | Transparent feature threshold scoring |
| Hugging Face model | Not measured by default | Depends on model download, hardware, and selected model |

The expected pattern is that the Random Forest model should perform strongly on the small demonstration dataset because the synthetic phishing examples contain clear suspicious indicators. This result should be discussed carefully. Strong performance on a small synthetic dataset does not prove that the system would perform equally well on real phishing campaigns.

## 12. Discussion

PhishGuard-Lite demonstrates that a lightweight phishing detection prototype can be built from simple and interpretable components. URL features are fast to compute and do not require visiting potentially harmful websites. Classical machine learning models are easy to train and deploy locally. Feature selection helps identify which features are most informative and supports the goal of lightweight detection.

The hybrid design has practical advantages. If the Hugging Face model is available, it can provide an additional pretrained text-based signal. If it is unavailable, the system still works. This is important in student environments where internet access, model cache availability, and hardware resources may be inconsistent.

False positives and false negatives must be considered carefully. A false positive occurs when a legitimate URL is classified as phishing. This can frustrate users and reduce confidence in the system. A false negative occurs when a phishing URL is classified as legitimate. This is more dangerous because it may expose a user to credential theft or malware. In a real security setting, the threshold could be adjusted depending on whether the priority is reducing false positives or reducing false negatives.

The project also shows the difference between explainability and interpretability. Random Forest feature importance gives a global view of useful features, while the rule-based explanation layer gives local reasons for one URL. These explanations are useful for presentation and education, but they should not be confused with rigorous causal explanations.

## 13. Limitations

The project has several important limitations:

- The included dataset is small and partly synthetic, so it is not a production benchmark.
- The system uses URL-only detection and does not inspect webpage content.
- The Hugging Face model may be unavailable without internet access or sufficient local resources.
- The system does not perform live crawling, JavaScript execution, screenshot comparison, or form analysis.
- Rule-based explanations are understandable but are not full XAI methods.
- The feature list does not include WHOIS data, domain age, DNS records, certificate metadata, or hosting reputation.
- Attackers can design URLs that avoid many obvious suspicious features.

These limitations are acceptable for a university prototype, but they must be acknowledged when interpreting the results.

## 14. Future Work

Several improvements would make PhishGuard-Lite more realistic and useful for future study. A browser extension could allow users to test links directly while browsing. Live webpage text extraction could add content-based features, such as page title, visible text, form fields, and login prompts. WHOIS and domain age features could help identify newly registered suspicious domains. DNS, certificate, and hosting metadata could provide additional security context.

Future experiments should also use larger public benchmark datasets and more careful cross-validation. Adversarial URL testing would be useful because attackers may intentionally design URLs that avoid obvious suspicious indicators. Finally, a stronger explainability layer could be added using SHAP or other local explanation methods, although this should be balanced against the project's lightweight design goal.

## 15. Conclusion

This report presented PhishGuard-Lite, a lightweight hybrid phishing URL detection system. The project combines handcrafted URL feature extraction, feature selection, classical machine learning, optional pretrained Hugging Face models, hybrid scoring, and simple explanations. It is designed as a practical adaptation of research ideas around phishing detection, feature selection, and pretrained models, rather than a full reproduction of a specific paper.

The main value of the project is educational. It provides a runnable system that demonstrates how URL features can be converted into model inputs, how lightweight models can be trained, how feature selection can be evaluated, how pretrained models can be integrated with fallback behavior, and how results can be explained in a classroom setting. The system is not a replacement for production phishing protection, but it is a clear foundation for further experimentation.

## 16. References

1. Scikit-learn Developers. *Scikit-learn: Machine Learning in Python*. https://scikit-learn.org/
2. Hugging Face. *Transformers Documentation*. https://huggingface.co/docs/transformers/
3. Streamlit. *Streamlit Documentation*. https://docs.streamlit.io/
4. OWASP. *Phishing and Social Engineering Security Guidance*. https://owasp.org/
5. NIST. *Cybersecurity Framework*. https://www.nist.gov/cyberframework
6. Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5-32.
7. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

## Classroom Presentation Section

### Demo steps

1. Open the Streamlit app with `streamlit run app.py`.
2. Start with the single URL tab.
3. Enter a clearly synthetic suspicious URL such as `https://paypal.verify-account.example-login.test/security/update`.
4. Show the final prediction, risk level, and confidence score.
5. Explain the model comparison table: classical ML score, Hugging Face score, rule-based score, and final hybrid score.
6. Show the explanation list and extracted feature table.
7. Move to batch evaluation and upload `data/sample_urls.csv`.
8. Show the confusion matrix and metrics.
9. Show the feature importance chart in the model details tab.

### What to show first

Start with the single URL analysis because it is easiest for the audience to understand. The teacher can immediately see the input, prediction, risk level, and explanation. After that, move to the dataset evaluation to show that the project also supports machine learning metrics.

### What results to explain

Explain accuracy, precision, recall, F1-score, true positive rate, false positive rate, confusion matrix, feature importance, and latency. Emphasize that recall is important because missed phishing URLs are risky, while precision is important because excessive false alarms reduce user trust.

### Likely teacher questions and short answers

**Question:** Is this a full reproduction of the referenced paper?  
**Answer:** No. It is a practical adaptation. It uses the same broad ideas of phishing detection, feature selection, machine learning, and pretrained models, but it does not use the exact dataset or experimental setup from the paper.

**Question:** Why use URL features instead of webpage content?  
**Answer:** URL features are lightweight, fast, and safer to extract because the system does not need to visit suspicious websites.

**Question:** Why is Random Forest the main model?  
**Answer:** It works well with structured numeric features, captures nonlinear patterns, and provides feature importance for explanation.

**Question:** What happens if the Hugging Face model fails to load?  
**Answer:** The app continues running with the classical ML model and rule-based layer using the fallback hybrid formula.

**Question:** Are the explanations full explainable AI?  
**Answer:** No. They are rule-based explanations for readability. They help users understand suspicious indicators, but they are not a complete XAI method.

**Question:** Can this detect all phishing URLs?  
**Answer:** No. It is a prototype. Real phishing detection requires larger datasets, live threat intelligence, content analysis, and continuous updates.
