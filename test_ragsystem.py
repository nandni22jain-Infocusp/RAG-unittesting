import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from typing import List

# Add the directory containing your RAG module to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your RAG class
from rag_system import SimpleRAG, logger

class TestSimpleRAG(unittest.TestCase):
    """Unit tests for SimpleRAG class"""
    
    def setUp(self):
        """Setup test fixtures"""
        # Mock the heavy dependencies to avoid actual model loading
        with patch('rag_system.HuggingFaceEmbedding'), \
             patch('rag_system.HuggingFaceLLM'):
            
            # Import here to avoid import errors during patching
            from rag_system import SimpleRAG
            self.rag = SimpleRAG()
            
            # Mock the models
            self.rag.embed_model = Mock()
            self.rag.llm = Mock()
    
    def test_01_initialization(self):
        """Test RAG system initialization"""
        self.assertIsNotNone(self.rag.embedding_model_name)
        self.assertIsNotNone(self.rag.llm_model_name)
        self.assertEqual(self.rag.documents, [])
        self.assertIsNone(self.rag.index)
        self.assertIsNone(self.rag.query_engine)
    
    @patch('rag_system.load_dataset')
    @patch('rag_system.Document')
    def test_02_load_documents_from_hf_success(self, mock_document, mock_load_dataset):
        """Test successful document loading from Hugging Face"""
        # Mock dataset
        mock_dataset = [
            {"text": "This is document 1"},
            {"text": "This is document 2"},
            {"text": "This is document 3"}
        ]
        mock_load_dataset.return_value = mock_dataset
        
        # Mock Document creation
        mock_doc_instances = [Mock(), Mock(), Mock()]
        mock_document.side_effect = mock_doc_instances
        
        # Test document loading
        result = self.rag.load_documents_from_hf("test_dataset", limit=3)
        
        # Assertions
        mock_load_dataset.assert_called_once_with("test_dataset", split="train")
        self.assertEqual(len(result), 3)
        self.assertEqual(len(self.rag.documents), 3)
        self.assertEqual(mock_document.call_count, 3)
    
    @patch('rag_system.load_dataset')
    def test_03_load_documents_limit_respected(self, mock_load_dataset):
        """Test that document limit is respected"""
        # Create more documents than limit
        mock_dataset = [{"text": f"Document {i}"} for i in range(10)]
        mock_load_dataset.return_value = mock_dataset
        
        with patch('rag_system.Document') as mock_document:
            mock_document.return_value = Mock()
            
            result = self.rag.load_documents_from_hf("test_dataset", limit=5)
            
            self.assertEqual(len(result), 5)
            self.assertEqual(mock_document.call_count, 5)
    
    @patch('rag_system.load_dataset')
    def test_04_load_documents_empty_text_filtered(self, mock_load_dataset):
        """Test that documents with empty text are filtered out"""
        mock_dataset = [
            {"text": "Valid document"},
            {"text": ""},  # Empty text
            {"text": None},  # None text
            {"text": "Another valid document"}
        ]
        mock_load_dataset.return_value = mock_dataset
        
        with patch('rag_system.Document') as mock_document:
            mock_document.return_value = Mock()
            
            result = self.rag.load_documents_from_hf("test_dataset", limit=10)
            
            # Should only load documents with valid text
            self.assertEqual(len(result), 2)
            self.assertEqual(mock_document.call_count, 2)
    
    @patch('rag_system.VectorStoreIndex')
    def test_05_create_index_success(self, mock_index_class):
        """Test successful index creation"""
        # Setup mock documents
        self.rag.documents = [Mock(), Mock()]
        mock_index = Mock()
        mock_index_class.from_documents.return_value = mock_index
        
        result = self.rag.create_index()
        
        # Assertions
        mock_index_class.from_documents.assert_called_once_with(
            self.rag.documents,
            embed_model=self.rag.embed_model,
            llm=self.rag.llm
        )
        logger.info("Vector index created test RESULT: {result}")
        logger.info("Vector index created test RAG index: {self.rag.index}")
        logger.info("Vector index created test mock index: {mock_index}")

        self.assertEqual(result, mock_index)
        self.assertEqual(self.rag.index, mock_index)
    
    def test_06_create_index_no_documents(self):
        """Test index creation fails when no documents are loaded"""
        self.rag.documents = []
        
        with self.assertRaises(ValueError) as context:
            self.rag.create_index()
        
        self.assertIn("No documents loaded", str(context.exception))
    
    @patch('rag_system.VectorIndexRetriever')
    def test_07_setup_retriever_success(self, mock_retriever_class):
        """Test successful retriever setup"""
        # Setup mock index
        self.rag.index = Mock()
        mock_retriever = Mock()
        mock_retriever_class.return_value = mock_retriever
        
        result = self.rag.setup_retriever(similarity_top_k=5)
        
        # Assertions
        mock_retriever_class.assert_called_once_with(
            index=self.rag.index,
            similarity_top_k=5
        )
        self.assertEqual(result, mock_retriever)
        self.assertEqual(self.rag.retriever, mock_retriever)
    
    def test_08_setup_retriever_no_index(self):
        """Test retriever setup fails when no index exists"""
        self.rag.index = None
        
        with self.assertRaises(ValueError) as context:
            self.rag.setup_retriever()
        
        self.assertIn("Index not created", str(context.exception))
    
    @patch('rag_system.SimilarityPostprocessor')
    @patch('rag_system.RetrieverQueryEngine')
    def test_09_setup_query_engine_success(self, mock_query_engine_class, mock_postprocessor_class):
        """Test successful query engine setup"""
        # Setup mock retriever
        self.rag.retriever = Mock()
        mock_postprocessor = Mock()
        mock_postprocessor_class.return_value = mock_postprocessor
        mock_query_engine = Mock()
        mock_query_engine_class.return_value = mock_query_engine
        
        result = self.rag.setup_query_engine()
        
        # Assertions
        mock_postprocessor_class.assert_called_once_with(similarity_cutoff=0.7)
        mock_query_engine_class.assert_called_once_with(
            retriever=self.rag.retriever,
            node_postprocessors=[mock_postprocessor]
        )
        self.assertEqual(result, mock_query_engine)
        self.assertEqual(self.rag.query_engine, mock_query_engine)
    
    def test_10_setup_query_engine_no_retriever(self):
        """Test query engine setup fails when no retriever exists"""
        # Ensure retriever is not set
        if hasattr(self.rag, 'retriever'):
            delattr(self.rag, 'retriever')
        
        with self.assertRaises(ValueError) as context:
            self.rag.setup_query_engine()
        
        self.assertIn("Retriever not setup", str(context.exception))
    
    def test_11_query_success(self):
        """Test successful query processing"""
        # Setup mock query engine
        self.rag.query_engine = Mock()
        mock_response = Mock()
        mock_response.__str__ = Mock(return_value="Test response")
        self.rag.query_engine.query.return_value = mock_response
        
        result = self.rag.query("Test question")
        
        # Assertions
        self.rag.query_engine.query.assert_called_once_with("Test question")
        self.assertEqual(result, "Test response")
    
    def test_12_query_no_query_engine(self):
        """Test query fails when no query engine exists"""
        self.rag.query_engine = None
        
        with self.assertRaises(ValueError) as context:
            self.rag.query("Test question")
        
        self.assertIn("Query engine not setup", str(context.exception))
    
    def test_13_get_retrieved_documents_success(self):
        """Test successful document retrieval"""
        # Setup mock retriever
        self.rag.retriever = Mock()
        
        # Mock retrieved nodes
        mock_node1 = Mock()
        mock_node1.node = Mock()
        mock_node2 = Mock()
        mock_node2.node = Mock()
        
        self.rag.retriever.retrieve.return_value = [mock_node1, mock_node2]
        
        result = self.rag.get_retrieved_documents("Test question")
        
        # Assertions
        self.rag.retriever.retrieve.assert_called_once_with("Test question")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], mock_node1.node)
        self.assertEqual(result[1], mock_node2.node)
    
    def test_14_get_retrieved_documents_no_retriever(self):
        """Test document retrieval fails when no retriever exists"""
        # Ensure retriever is not set
        if hasattr(self.rag, 'retriever'):
            delattr(self.rag, 'retriever')
        
        with self.assertRaises(ValueError) as context:
            self.rag.get_retrieved_documents("Test question")
        
        self.assertIn("Retriever not setup", str(context.exception))


class TestRAGWorkflow(unittest.TestCase):
    """Integration tests for RAG workflow"""
    
    def setUp(self):
        """Setup test fixtures"""
        with patch('rag_system.HuggingFaceEmbedding'), \
             patch('rag_system.HuggingFaceLLM'):
            
            from rag_system import SimpleRAG
            self.rag = SimpleRAG()
            self.rag.embed_model = Mock()
            self.rag.llm = Mock()
    
    @patch('rag_system.load_dataset')
    @patch('rag_system.Document')
    @patch('rag_system.VectorStoreIndex')
    @patch('rag_system.VectorIndexRetriever')
    @patch('rag_system.SimilarityPostprocessor')
    @patch('rag_system.RetrieverQueryEngine')
    def test_full_rag_workflow(self, mock_query_engine_class, mock_postprocessor_class,
                              mock_retriever_class, mock_index_class, mock_document, mock_load_dataset):
        """Test complete RAG workflow from document loading to querying"""
        
        # Mock dataset loading
        mock_dataset = [{"text": "Document 1"}, {"text": "Document 2"}]
        mock_load_dataset.return_value = mock_dataset
        mock_document.return_value = Mock()
        
        # Mock index creation
        mock_index = Mock()
        mock_index_class.from_documents.return_value = mock_index
        
        # Mock retriever setup
        mock_retriever = Mock()
        mock_retriever_class.return_value = mock_retriever
        
        # Mock query engine setup
        mock_postprocessor = Mock()
        mock_postprocessor_class.return_value = mock_postprocessor
        mock_query_engine = Mock()
        mock_query_engine_class.return_value = mock_query_engine
        
        # Mock query response
        mock_response = Mock()
        mock_response.__str__ = Mock(return_value="Final response")
        mock_query_engine.query.return_value = mock_response
        
        # Execute full workflow
        documents = self.rag.load_documents_from_hf("test_dataset", limit=2)
        index = self.rag.create_index()
        retriever = self.rag.setup_retriever()
        query_engine = self.rag.setup_query_engine()
        response = self.rag.query("Test question")
        
        # Verify workflow
        self.assertEqual(len(documents), 2)
        self.assertIsNotNone(index)
        self.assertIsNotNone(retriever)
        self.assertIsNotNone(query_engine)
        self.assertEqual(response, "Final response")
        
        # Verify all components were called
        mock_load_dataset.assert_called_once()
        mock_index_class.from_documents.assert_called_once()
        mock_retriever_class.assert_called_once()
        mock_query_engine_class.assert_called_once()
        mock_query_engine.query.assert_called_once_with("Test question")


class TestRAGErrorHandling(unittest.TestCase):
    """Test error handling in RAG system"""
    
    def setUp(self):
        """Setup test fixtures"""
        with patch('rag_system.HuggingFaceEmbedding'), \
             patch('rag_system.HuggingFaceLLM'):
            
            from rag_system import SimpleRAG
            self.rag = SimpleRAG()
            self.rag.embed_model = Mock()
            self.rag.llm = Mock()
    
    @patch('rag_system.load_dataset')
    def test_dataset_loading_error(self, mock_load_dataset):
        """Test error handling when dataset loading fails"""
        mock_load_dataset.side_effect = Exception("Dataset not found")
        
        with self.assertRaises(Exception) as context:
            self.rag.load_documents_from_hf("invalid_dataset")
        
        self.assertIn("Dataset not found", str(context.exception))
    
    @patch('rag_system.VectorStoreIndex')
    def test_index_creation_error(self, mock_index_class):
        """Test error handling when index creation fails"""
        self.rag.documents = [Mock()]
        mock_index_class.from_documents.side_effect = Exception("Index creation failed")
        
        with self.assertRaises(Exception) as context:
            self.rag.create_index()
        
        self.assertIn("Index creation failed", str(context.exception))
    
    def test_query_with_invalid_engine(self):
        """Test query with invalid query engine"""
        self.rag.query_engine = Mock()
        self.rag.query_engine.query.side_effect = Exception("Query processing failed")
        
        with self.assertRaises(Exception) as context:
            self.rag.query("Test question")
        
        self.assertIn("Query processing failed", str(context.exception))


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSimpleRAG))
    suite.addTests(loader.loadTestsFromTestCase(TestRAGWorkflow))
    suite.addTests(loader.loadTestsFromTestCase(TestRAGErrorHandling))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    # Run tests
    print("Running RAG System Unit Tests...")
    print("=" * 50)
    
    result = run_tests()
    
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")