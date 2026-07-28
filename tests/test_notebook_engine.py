import unittest
import numpy as np
import cv2

from worksheet_ocr import (
    WorksheetOCRPipeline,
    PipelineConfig,
    ImagePreprocessor,
    DocumentLayoutAnalyzer,
    DocumentUnderstandingLayer,
    SubjectClassifier,
    GrammarAnalyzer,
    VocabularyAnalyzer,
    ContextualSpellChecker,
    NotebookEvaluator,
    MissingInfoDetector,
    AnomalyDetector,
    MistakeClassifier,
    RepeatedMistakeTracker,
    LearningAnalyticsModule,
    FlashcardGenerator,
    TeacherFeedbackGenerator
)

class TestAINotebookIntelligenceEngine(unittest.TestCase):

    def setUp(self):
        self.config = PipelineConfig()
        self.test_img = np.full((400, 500, 3), 245, dtype=np.uint8)
        cv2.putText(self.test_img, "Worksheet 1: Mathematics", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)
        cv2.putText(self.test_img, "Question 1. Calculate the area of a square with side 5 cm.", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1)
        cv2.putText(self.test_img, "Area = 25 cm2", (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 2)

    def test_document_detection_and_preprocessing(self):
        preprocessor = ImagePreprocessor(self.config)
        processed, meta = preprocessor.process(self.test_img)
        self.assertIsNotNone(processed)
        self.assertIn("shadow_reduction_applied", meta)
        self.assertIn("paper_page_detected", meta)

    def test_document_layout_analysis(self):
        analyzer = DocumentLayoutAnalyzer(self.config)
        mock_regions = [
            {"bbox": [30, 20, 300, 50], "extracted_text": "Worksheet 1: Mathematics", "region_type": "printed", "confidence": 0.95},
            {"bbox": [30, 80, 450, 110], "extracted_text": "Question 1. Calculate area of square side 5 cm.", "region_type": "printed", "confidence": 0.92},
            {"bbox": [30, 140, 250, 180], "extracted_text": "25 cm2", "region_type": "handwritten", "confidence": 0.88}
        ]
        layout = analyzer.analyze_layout(self.test_img, mock_regions)
        self.assertIn("layout_elements", layout)
        self.assertIn("hierarchy", layout)
        self.assertEqual(len(layout["layout_elements"]), 3)
        self.assertEqual(layout["layout_elements"][0]["role"], "title")
        self.assertEqual(layout["layout_elements"][1]["role"], "question_number")

    def test_document_understanding_layer(self):
        du = DocumentUnderstandingLayer(self.config)
        layout_data = {
            "layout_elements": [
                {"element_id": 1, "role": "question_number", "text": "Q1. Calculate area."},
                {"element_id": 2, "role": "answer", "text": "25 cm2"}
            ]
        }
        struct = du.process_document_structure(layout_data)
        self.assertIn("qa_pairs", struct)
        self.assertEqual(len(struct["qa_pairs"]), 1)
        self.assertEqual(struct["qa_pairs"][0]["answer_text"], "25 cm2")

    def test_subject_classifier(self):
        sc = SubjectClassifier(self.config)
        res = sc.identify_subject("Calculate the hypotenuse triangle area x + y = 10", [])
        self.assertEqual(res["identified_subject"], "Mathematics")

    def test_grammar_analyzer(self):
        ga = GrammarAnalyzer(self.config)
        res = ga.analyze_text("they is going to school. the the book was good")
        self.assertIn("detected_issues", res)
        self.assertTrue(len(res["detected_issues"]) > 0)
        self.assertLess(res["grammar_score"], 100.0)

    def test_vocabulary_analyzer(self):
        va = VocabularyAnalyzer(self.config)
        res = va.analyze_vocabulary("the big thing had a good effect on their work")
        self.assertIn("suggestions", res)
        self.assertTrue(len(res["suggestions"]) > 0)

    def test_spell_checker(self):
        sc = ContextualSpellChecker(self.config)
        res = sc.check_spelling("We need to recieve the seperated answers becuase of testing.")
        self.assertIn("spelling_errors", res)
        self.assertTrue(len(res["spelling_errors"]) >= 2)

    def test_notebook_evaluator(self):
        ne = NotebookEvaluator(self.config)
        qa_pairs = [{
            "question_id": 1,
            "question_text": "Find the area of a rectangle with length 4 cm and width 3 cm.",
            "answer_text": "12"  # Missing unit
        }]
        res = ne.evaluate_notebook(qa_pairs, "Mathematics")
        self.assertIn("question_evaluations", res)
        self.assertIn("Missing measurement unit", res["question_evaluations"][0]["explanation"])

    def test_missing_info_detector(self):
        mid = MissingInfoDetector(self.config)
        qa_pairs = [{
            "question_id": 1,
            "question_text": "Explain the process of photosynthesis.",
            "answer_text": "" # Unanswered
        }]
        missing = mid.detect_missing_info(qa_pairs, {})
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["type"], "unanswered_question")

    def test_anomaly_detector(self):
        ad = AnomalyDetector(self.config)
        layout_data = {
            "blank_spaces": [{"y_start": 100, "y_end": 300, "height_px": 200}]
        }
        anomalies = ad.detect_anomalies([], layout_data)
        self.assertTrue(len(anomalies) > 0)

    def test_flashcard_generator(self):
        fg = FlashcardGenerator(self.config)
        mistakes = [
            {
                "category": "spelling",
                "original_text": "recieve",
                "suggested_correction": "receive",
                "explanation": "i before e except after c"
            }
        ]
        flashcards = fg.generate_flashcards(mistakes, "English")
        self.assertEqual(len(flashcards), 1)
        self.assertIn("Spelling Revision", flashcards[0]["title"])

    def test_teacher_feedback_generator(self):
        tfg = TeacherFeedbackGenerator(self.config)
        analytics = {"subject_mastery_score": 85.0, "grammar_score": 80.0}
        mistakes = [{"category": "missing_unit", "original_text": "15"}]
        feedback = tfg.generate_teacher_feedback(analytics, mistakes, "Mathematics")
        self.assertIn("teacher_summary_comment", feedback)
        self.assertIn("prioritized_suggestions", feedback)

    def test_end_to_end_pipeline_integration(self):
        pipeline = WorksheetOCRPipeline(self.config)
        result = pipeline.process_image(self.test_img, page_num=1)
        
        # Verify required structured JSON output sections
        self.assertIn("layout_understanding", result)
        self.assertIn("document_structure", result)
        self.assertIn("subject_identification", result)
        self.assertIn("grammar_analysis", result)
        self.assertIn("vocabulary_analysis", result)
        self.assertIn("spell_checking", result)
        self.assertIn("concept_evaluation", result)
        self.assertIn("missing_information", result)
        self.assertIn("anomalies", result)
        self.assertIn("categorized_mistakes", result)
        self.assertIn("repeated_mistake_analysis", result)
        self.assertIn("learning_analytics", result)
        self.assertIn("personalized_flashcards", result)
        self.assertIn("teacher_feedback", result)

if __name__ == "__main__":
    unittest.main()
