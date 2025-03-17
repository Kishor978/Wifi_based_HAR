from gevent import monkey
monkey.patch_all()

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import json
import logging
import time
from datetime import datetime
import os
import csv
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')
PREDICTION_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'active_ap', 'predictions.csv')

ACTIVITY_MAPPING = {
    0: "Walking",
    1: "Standing"
}

# Offline detection settings
OFFLINE_THRESHOLD_SECONDS = 120
@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    emit('connection_response', {'status': 'connected', 'is_live': False})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

def check_prediction_file():
    """Continuously check the prediction.csv file for updates and broadcast changes"""
    last_modified_time = 0
    last_prediction = None
    last_value_change_time = time.time()
    
    logger.info(f"Starting prediction file monitoring: {PREDICTION_FILE}")
    
    while True:
        try:
            if os.path.exists(PREDICTION_FILE):
                current_modified_time = os.path.getmtime(PREDICTION_FILE)
                if current_modified_time > last_modified_time:
                    last_modified_time = current_modified_time
                    
                    # Read the latest prediction
                    try:
                        df = pd.read_csv(PREDICTION_FILE)
                        if not df.empty:
                            latest_row = df.iloc[-1]
                            prediction_value = int(latest_row.iloc[0])
                            
                            # Check if prediction has changed
                            if prediction_value != last_prediction:
                                last_value_change_time = time.time()
                                last_prediction = prediction_value
                                activity = ACTIVITY_MAPPING.get(prediction_value, "Unknown")
                                
                                # Create confidence scores (100% for the detected activity)
                                confidence_scores = [0, 0]
                                confidence_scores[prediction_value] = 100
                                dashboard_data = {
                                    'hypothesis': activity,
                                    'classnames': list(ACTIVITY_MAPPING.values()),
                                    'confidence_scores': confidence_scores,
                                    'duration': 0,
                                    'timestamp': datetime.now().isoformat(),
                                    'is_offline': False
                                }

                                logger.info(f"Broadcasting new prediction: {activity}")
                                socketio.emit('activity_update', dashboard_data)
                                socketio.emit('live_status', {'is_live': True})
                    except Exception as e:
                        logger.error(f"Error reading prediction file: {str(e)}")
            
            # Check if we've been receiving the same value for too long
            current_time = time.time()
            time_since_last_change = current_time - last_value_change_time
            
            if time_since_last_change > OFFLINE_THRESHOLD_SECONDS:
                dashboard_data = {
                    'hypothesis': "Offline",
                    'classnames': list(ACTIVITY_MAPPING.values()),
                    'confidence_scores': [0, 0],
                    'duration': 0,
                    'timestamp': datetime.now().isoformat(),
                    'is_offline': True
                }

                if int(time_since_last_change) % 5 == 0:
                    logger.info(f"System appears to be offline (no value change for {int(time_since_last_change)} seconds)")
                    socketio.emit('activity_update', dashboard_data)
                    socketio.emit('live_status', {'is_live': False})
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error in prediction monitoring: {str(e)}")
            time.sleep(1)

def main(args):
    logger.info(f"Starting server on http://{args.host}:{args.port}")
    from threading import Thread
    prediction_thread = Thread(target=check_prediction_file, daemon=True)
    prediction_thread.start()
    
    socketio.run(app, host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HAR Dashboard Server")
    
    parser.add_argument("--host", type=str, default="0.0.0.0",
                      help="Host to run the dashboard server on")
    parser.add_argument("--port", type=int, default=5000,
                      help="Port to run the dashboard server on")
    parser.add_argument("--debug", action="store_true",
                      help="Run in debug mode")
    
    args = parser.parse_args()
    logger.info("Starting HAR Dashboard Server...")
    main(args)
