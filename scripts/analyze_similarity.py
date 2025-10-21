"""disease similarity analysis"""

import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.data_loader import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def analyze_disease_similarity():    
    logging.info("starting disease similarity analysis")
    
    data_loader = DataLoader(Path("data/symptom_matrix.csv"))
    df, symptom_cols = data_loader.load_matrix()
    
    logging.info(f"original data: {len(df)} rows, {len(symptom_cols)} symptoms")
    
    original_count = len(df)
    df_aggregated = df.groupby('diseases')[symptom_cols].max().reset_index()
    unique_count = len(df_aggregated)
    
    logging.info(f"after aggregation: {unique_count} unique diseases")
    logging.info(f"each disease represented by full symptom profile")
    
    disease_descriptions = []
    disease_names = []
    
    for _, row in df_aggregated.iterrows():
        symptoms = [col for col in symptom_cols if row[col] == 1]
        description = ' '.join(symptoms)
        disease_descriptions.append(description)
        disease_names.append(row['diseases'])
    
    logging.info("created disease descriptions from symptoms")
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(disease_descriptions)
    
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    most_similar_pairs = []
    for i in range(len(disease_names)):
        for j in range(i+1, len(disease_names)):
            similarity = similarity_matrix[i][j]
            most_similar_pairs.append({
                'disease1': disease_names[i],
                'disease2': disease_names[j], 
                'similarity': similarity
            })
    
    most_similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
    
    similarities = [pair['similarity'] for pair in most_similar_pairs]
    avg_similarity = np.mean(similarities)
    max_similarity = np.max(similarities)
    min_similarity = np.min(similarities)
    
    high_similarity_threshold = 0.8
    medium_similarity_threshold = 0.5
    
    high_similarity_count = sum(1 for s in similarities if s > high_similarity_threshold)
    medium_similarity_count = sum(1 for s in similarities if s > medium_similarity_threshold)
    total_pairs = len(most_similar_pairs)
    
    print("\n" + "="*60)
    print("tf-idf and cosine similarity analysis")
    print("="*60)
    
    print(f"analyzed: {len(disease_names)} diseases")
    print(f"disease pairs: {total_pairs}")
    print(f"average similarity: {avg_similarity:.3f}")
    print(f"max similarity: {max_similarity:.3f}")
    print(f"min similarity: {min_similarity:.3f}")
    
    print(f"\nsimilarity distribution:")
    print(f"pairs with similarity > {high_similarity_threshold}: {high_similarity_count}/{total_pairs} ({100*high_similarity_count/total_pairs:.1f}%)")
    print(f"pairs with similarity > {medium_similarity_threshold}: {medium_similarity_count}/{total_pairs} ({100*medium_similarity_count/total_pairs:.1f}%)")
    
    print(f"\nmost similar disease pairs:")
    for i, pair in enumerate(most_similar_pairs[:15], 1):
        print(f"{i:2}. {pair['disease1'][:40]:<40} <-> {pair['disease2'][:40]:<40}: {pair['similarity']:.3f}")
    
    print(f"\n" + "="*60)
    print("conclusions")
    print("="*60)
    
    print(f"\ndetailed stats:")
    print(f"- very different (0.0-0.3): {sum(1 for s in similarities if s <= 0.3)}")
    print(f"- somewhat different (0.3-0.5): {sum(1 for s in similarities if 0.3 < s <= 0.5)}")
    print(f"- moderately similar (0.5-0.8): {sum(1 for s in similarities if 0.5 < s <= 0.8)}")
    print(f"- highly similar (0.8-1.0): {sum(1 for s in similarities if s > 0.8)}")
    
    return {
        'most_similar_pairs': most_similar_pairs[:10],
        'stats': {
            'total_diseases': len(disease_names),
            'total_pairs': total_pairs,
            'avg_similarity': avg_similarity,
            'max_similarity': max_similarity,
            'min_similarity': min_similarity,
            'high_similarity_count': high_similarity_count,
            'high_similarity_percentage': 100*high_similarity_count/total_pairs
        }
    }

def analyze_practical_performance():
    """test chatbot performance on different symptoms."""
    
    print(f"\n" + "="*60)
    print("practical chatbot performance tests")
    print("="*60)
    
    data_loader = DataLoader(Path("data/symptom_matrix.csv"))
    
    df, symptom_cols = data_loader.load_matrix()
    df_aggregated = df.groupby('diseases')[symptom_cols].max().reset_index()
    
    print(f"testing on {len(df_aggregated)} unique diseases")
    
    test_cases = [
        {"name": "flu symptoms", "symptoms": ["fever", "cough", "headache", "fatigue"]},
        {"name": "heart problems", "symptoms": ["sharp chest pain", "shortness of breath"]},
        {"name": "stomach issues", "symptoms": ["nausea", "vomiting", "abdominal pain"]},
        {"name": "skin symptoms", "symptoms": ["skin rash", "itching of skin"]},
        {"name": "neurological problems", "symptoms": ["dizziness", "headache", "confusion"]},
    ]
    
    results = []
    for test_case in test_cases:
        try:
            matches = data_loader.find_diseases_by_symptoms(
                test_case["symptoms"], min_hits=2, top_k=5
            )
            
            print(f"\n{test_case['name']} - symptoms: {test_case['symptoms']}")
            if matches:
                print(f"  found {len(matches)} diseases:")
                for i, match in enumerate(matches, 1):
                    print(f"    {i}. {match['disease']} (score: {match['score']}, "
                          f"matched: {match['matched_symptoms']})")
            else:
                print("  no matches found")
                
            results.append({
                'test_name': test_case['name'],
                'input_symptoms': test_case['symptoms'],
                'found_count': len(matches),
                'top_diseases': [m['disease'] for m in matches[:3]]
            })
            
        except Exception as e:
            print(f"  error: {e}")
    
    return results

def check_available_symptoms():
    """check what symptoms are available in database."""
    data_loader = DataLoader(Path("data/symptom_matrix.csv"))
    df, symptom_cols = data_loader.load_matrix()
    
    print("\n" + "="*60)
    print("available symptoms in database")
    print("="*60)
    print(f"total symptoms: {len(symptom_cols)}")
    
    print("\nexample symptoms (first 20):")
    for i, symptom in enumerate(sorted(symptom_cols)[:20], 1):
        print(f"{i:2}. {symptom}")
    
    missing_symptoms = []
    search_terms = [
        "chest pain", "shortness of breath", "itching", "skin rash"
    ]
    
    print("\n" + "="*40)
    print("checking specific symptoms:")
    print("="*40)
    
    for term in search_terms:
        if term in symptom_cols:
            print(f"found: '{term}'")
        else:
            missing_symptoms.append(term)
            similar = [s for s in symptom_cols if term.replace("_", " ").replace(" ", "") in s.replace("_", "").replace(" ", "")]
            if similar:
                print(f"not found: '{term}', but maybe: {similar[:3]}")
            else:
                print(f"not found: '{term}'")
    
    return symptom_cols

if __name__ == "__main__":
    similarity_results = analyze_disease_similarity()
    performance_results = analyze_practical_performance()
    available_symptoms = check_available_symptoms()
    
    print(f"\n" + "="*60)
    print("analysis completed")
    print("="*60)