import asyncio
import unittest
from pathlib import Path
import logging
from typing import Dict, Any
import os

import openai
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.logging import RichHandler

from config_extractor import ConfigExtractor, DocumentProcessor, DocumentStructureAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("test_config_extractor")
console = Console()

class TestConfigExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Ensure OpenAI API key is set
        cls.api_key = "test_key"
        if not cls.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        # Initialize OpenAI client
        cls.openai_client = openai.OpenAI(api_key=cls.api_key)
        
        # Initialize LangChain ChatOpenAI
        cls.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0,
            openai_api_key=cls.api_key
        )
        
        # Initialize ConfigExtractor
        cls.config_extractor = ConfigExtractor(model_name="gpt-4")
        
        # Initialize DocumentProcessor
        cls.doc_processor = DocumentProcessor()

    async def test_openai_connectivity(self):
        """Test OpenAI API connectivity."""
        try:
            # Test simple completion
            response = await self.llm.agenerate([[{"role": "user", "content": "Hello"}]])
            self.assertIsNotNone(response)
            logger.info(response)
            logger.info("✅ OpenAI API connection successful")
        except Exception as e:
            logger.error(f"❌ OpenAI API connection failed: {str(e)}")
            raise

    async def test_document_processing_flow(self):
        """Test the document processing flow including first and second pass."""
        # Sample test document path
        test_doc_path = Path("test_document.docx")  # Replace with your test document
        
        if not test_doc_path.exists():
            logger.error(f"❌ Test document not found: {test_doc_path}")
            return
        
        try:
            # Test document reading
            text = await self.config_extractor._read_document(test_doc_path, "docx")
            self.assertIsNotNone(text)
            logger.info("✅ Document reading successful")

            # Test first pass analysis
            first_pass_results = await self.config_extractor._first_pass_analysis(text)
            self.assertIsNotNone(first_pass_results)
            self.assertIn("document_summary", first_pass_results)
            self.assertIn("potential_configs", first_pass_results)
            logger.info("✅ First pass analysis successful")

            # Test structure analysis
            structure_analysis = await self.config_extractor._analyze_structure(text)
            self.assertIsNotNone(structure_analysis)
            self.assertIn("sections", structure_analysis)
            self.assertIn("relationships", structure_analysis)
            logger.info("✅ Structure analysis successful")

            # Test related contexts
            related_contexts = await self.config_extractor._find_related_contexts(text)
            self.assertIsNotNone(related_contexts)
            logger.info("✅ Related contexts retrieval successful")

            # Test second pass extraction
            config_result = await self.config_extractor._second_pass_extraction(
                text,
                first_pass_results,
                structure_analysis,
                related_contexts
            )
            self.assertIsNotNone(config_result)
            logger.info("✅ Second pass extraction successful")

        except Exception as e:
            logger.error(f"❌ Document processing test failed: {str(e)}")
            raise

    async def test_document_structure_analyzer(self):
        """Test the DocumentStructureAnalyzer functionality."""
        analyzer = DocumentStructureAnalyzer()
        test_text = """
        Configuration Settings:
        host = localhost
        port = 8080
        
        Database Configuration:
        db_host = 127.0.0.1
        db_port = 5432
        
        API Settings:
        api_key = secret_key
        api_endpoint = /api/v1
        """
        
        try:
            # Test structure analysis
            structure = await analyzer.analyze_structure(test_text)
            self.assertIsNotNone(structure)
            self.assertIn("sections", structure)
            self.assertIn("relationships", structure)
            self.assertIn("config_sections", structure)
            
            # Verify sections are properly identified
            sections = structure["sections"]
            self.assertTrue(len(sections) > 0)
            self.assertTrue(any(section["type"] == "configuration" for section in sections))
            
            logger.info("✅ Document structure analyzer test successful")
            
        except Exception as e:
            logger.error(f"❌ Document structure analyzer test failed: {str(e)}")
            raise

    def test_processing_prerequisites(self):
        """Test if all prerequisites for document processing are met."""
        try:
            # Check if output directory exists
            self.assertTrue(hasattr(self.doc_processor.config, 'output_dir'))
            os.makedirs(self.doc_processor.config.output_dir, exist_ok=True)
            
            # Check if temp directory exists
            self.assertTrue(hasattr(self.doc_processor.config, 'temp_dir'))
            os.makedirs(self.doc_processor.config.temp_dir, exist_ok=True)
            
            # Check if log directory exists
            self.assertTrue(hasattr(self.doc_processor.config, 'log_dir'))
            os.makedirs(self.doc_processor.config.log_dir, exist_ok=True)
            
            logger.info("✅ Processing prerequisites test successful")
            
        except Exception as e:
            logger.error(f"❌ Processing prerequisites test failed: {str(e)}")
            raise

class TestDocumentProcessor(unittest.TestCase):
    """Test class specifically for DocumentProcessor functionality."""
    
    def setUp(self):
        """Set up test environment for each test."""
        self.processor = DocumentProcessor()
        self.test_dir = Path("input")
        self.test_dir.mkdir(exist_ok=True)

    async def test_process_directory(self):
        """Test directory processing functionality."""
        # Create test documents
        test_doc = self.test_dir / "LLD_GAVI_iSupplier_Outbound.docx"
        if not test_doc.exists():
            logger.warning(f"Test document not found: {test_doc}")
            return

        try:
            results = await self.processor.process_directory(self.test_dir)
            self.assertIsInstance(results, dict)
            logger.info("✅ Directory processing test successful")
        except Exception as e:
            logger.error(f"❌ Directory processing test failed: {str(e)}")
            raise

    async def test_process_file(self):
        """Test individual file processing."""
        test_doc = self.test_dir / "LLD_GAVI_iSupplier_Outbound.docx"
        if not test_doc.exists():
            logger.warning(f"Test document not found: {test_doc}")
            return

        try:
            from rich.progress import Progress
            with Progress() as progress:
                task_id = progress.add_task("Testing file processing")
                result = await self.processor.process_file(test_doc, "docx", progress, task_id)
                self.assertIsInstance(result, tuple)
                self.assertEqual(len(result), 2)
                logger.info("✅ File processing test successful")
        except Exception as e:
            logger.error(f"❌ File processing test failed: {str(e)}")

async def run_specific_test_class(test_class):
    """Run a specific test class with async support."""
    # Create a test loader
    # Create a test loader
    loader = unittest.TestLoader()
    # Load tests from the specified class
    suite = loader.loadTestsFromTestCase(test_class)
    # Create a test runner
    runner = unittest.TextTestRunner()
    # Run the tests using asyncio
    for test in suite:
        for method in [attr for attr in dir(test) if attr.startswith('test_')]:
            test_method = getattr(test, method)
            if asyncio.iscoroutinefunction(test_method):
                await test_method()
            else:
                test_method()
    # Run the tests
    runner.run(suite)

# Method 2: Creating test suite manually
async def run_multiple_test_classes(test_classes):
    """Run multiple test classes with async support."""
    # Create a test suite
    # Create a test suite
    suite = unittest.TestSuite()
    # Add each test class to the suite
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    # Run the suite with async support
    for test in suite:
        for method in [attr for attr in dir(test) if attr.startswith('test_')]:
            test_method = getattr(test, method)
            if asyncio.iscoroutinefunction(test_method):
                await test_method()
            else:
                test_method()
    # Run the suite
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == "__main__":
    print("Running chatgpt connectivity test:")
    asyncio.run(run_specific_test_class(TestConfigExtractor))
