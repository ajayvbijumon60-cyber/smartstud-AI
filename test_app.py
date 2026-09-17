import unittest
import json

from app import app

class SmartStudTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index_route(self):
        """Test main homepage loads correctly"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'SmartStud.Ai', response.data)
        self.assertIn(b'<!DOCTYPE html>', response.data)

    def test_chat_api_empty_prompt(self):
        """Test chat API with empty prompt"""
        response = self.app.post('/api/chat',
                                 data=json.dumps({'prompt': ''}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('reply', data)

    def test_chat_api_with_prompt(self):
        """Test chat API with valid prompt"""
        response = self.app.post('/api/chat',
                                 data=json.dumps({'prompt': 'What is photosynthesis?'}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('reply', data)

    def test_flashcards_api(self):
        """Test flashcards API"""
        response = self.app.post('/api/flashcards',
                                 data=json.dumps({'topic': 'Calculus'}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('cards', data)
        self.assertGreater(len(data['cards']), 0)

    def test_quiz_api_difficulties(self):
        """Test that quiz API returns distinct question sets for different difficulty levels"""
        # Beginner
        resp_beg = self.app.post('/api/quiz',
                                 data=json.dumps({'topic': 'Robotics', 'difficulty': 'beginner'}),
                                 content_type='application/json')
        self.assertEqual(resp_beg.status_code, 200)
        q_beg = json.loads(resp_beg.data).get('questions', [])

        # Intermediate
        resp_int = self.app.post('/api/quiz',
                                 data=json.dumps({'topic': 'Robotics', 'difficulty': 'intermediate'}),
                                 content_type='application/json')
        self.assertEqual(resp_int.status_code, 200)
        q_int = json.loads(resp_int.data).get('questions', [])

        # Advanced
        resp_adv = self.app.post('/api/quiz',
                                 data=json.dumps({'topic': 'Robotics', 'difficulty': 'advanced'}),
                                 content_type='application/json')
        self.assertEqual(resp_adv.status_code, 200)
        q_adv = json.loads(resp_adv.data).get('questions', [])

        self.assertGreater(len(q_beg), 0)
        self.assertGreater(len(q_int), 0)
        self.assertGreater(len(q_adv), 0)

        # Assert questions across difficulties are distinct
        self.assertNotEqual(q_beg[0]['q'], q_int[0]['q'])
        self.assertNotEqual(q_int[0]['q'], q_adv[0]['q'])

    def test_mindmap_api(self):
        """Test mindmap API"""
        response = self.app.post('/api/mindmap',
                                 data=json.dumps({'topic': 'World War II'}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('center', data)
        self.assertIn('branches', data)

    def test_course_api_with_detailed_content(self):
        """Test course API returns chapters containing both points and detailedContent"""
        response = self.app.post('/api/course',
                                 data=json.dumps({'topic': 'Robotics', 'chapters': 3}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('chapters', data)
        chapters = data.get('chapters', [])
        self.assertGreater(len(chapters), 0)

        for chap in chapters:
            self.assertIn('title', chap)
            self.assertIn('points', chap)
            self.assertIn('detailedContent', chap)
            self.assertGreater(len(chap['detailedContent']), 20)

    def test_room_create_and_get(self):
        """Test Google Meet style room creation and lookup"""
        resp = self.app.post('/api/room/create',
                             data=json.dumps({'topic': 'Physics Group Study', 'host_name': 'Alice'}),
                             content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data.get('success'))
        self.assertIn('roomCode', data)
        room_code = data['roomCode']
        
        # Test lookup
        resp_get = self.app.get(f'/api/room/{room_code}')
        self.assertEqual(resp_get.status_code, 200)
        data_get = json.loads(resp_get.data)
        self.assertEqual(data_get.get('roomCode'), room_code)
        self.assertEqual(data_get.get('topic'), 'Physics Group Study')

if __name__ == '__main__':
    unittest.main()
