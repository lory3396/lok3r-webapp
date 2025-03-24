import os, cv2, numpy as np, base64
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
from treys import Card, Evaluator, Deck

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='eventlet')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Carica il modello YOLO (verifica che il percorso sia corretto)
model = YOLO('models/Poker_cards.pt')

# Mappatura degli indici alle carte (modifica se necessario in base al tuo modello)
card_class_mapping = {
    0: "2C", 1: "2D", 2: "2H", 3: "2S",
    4: "3C", 5: "3D", 6: "3H", 7: "3S",
    8: "4C", 9: "4D", 10: "4H", 11: "4S",
    12: "5C", 13: "5D", 14: "5H", 15: "5S",
    16: "6C", 17: "6D", 18: "6H", 19: "6S",
    20: "7C", 21: "7D", 22: "7H", 23: "7S",
    24: "8C", 25: "8D", 26: "8H", 27: "8S",
    28: "9C", 29: "9D", 30: "9H", 31: "9S",
    32: "TC", 33: "TD", 34: "TH", 35: "TS",
    36: "JC", 37: "JD", 38: "JH", 39: "JS",
    40: "QC", 41: "QD", 42: "QH", 43: "QS",
    44: "KC", 45: "KD", 46: "KH", 47: "KS",
    48: "AC", 49: "AD", 50: "AH", 51: "AS"
}

def base64_to_image(base64_str):
    """Converte una stringa base64 in un'immagine OpenCV."""
    header, encoded = base64_str.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    return img

def image_to_base64(image):
    """Converte un'immagine OpenCV in una stringa base64."""
    retval, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return "data:image/jpeg;base64," + jpg_as_text

def detect_cards_from_image(image):
    """
    Esegue la rilevazione delle carte con YOLO e disegna i bounding box.
    Divide i rilevamenti in base alla posizione verticale:
      - Le carte con il centro nella parte inferiore (y_center ≥ 2/3 dell'altezza) sono considerate hole cards (tua mano).
      - Le carte con il centro nella parte centrale (1/3 ≤ y_center < 2/3 dell'altezza) sono considerate board.
    """
    results = model(image)
    predictions = []
    height, width, _ = image.shape

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = int(box.cls[0])
                card_label = card_class_mapping.get(cls, "Unknown")
                # Calcola il centro della bounding box
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                predictions.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'label': card_label,
                    'center': (center_x, center_y)
                })
                # Disegna il bounding box e la label sull'immagine
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.putText(image, card_label, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    # Separa i rilevamenti in base alla posizione verticale
    player_cards = []  # Hole cards: carte nella parte inferiore
    board_cards = []   # Board: carte nella parte centrale
    for pred in predictions:
        center_y = pred['center'][1]
        if center_y >= (2/3.0) * height:
            player_cards.append(pred)
        elif center_y >= (1/3.0) * height and center_y < (2/3.0) * height:
            board_cards.append(pred)
    return predictions, image, player_cards, board_cards

def monte_carlo_simulation(player_cards, board_cards, simulations=25000):
    """
    Esegue una simulazione Monte Carlo per valutare la forza della mano.
    Vengono considerate le hole cards (player_cards) e le board cards già note.
    Se il board è parziale, vengono simulate le carte mancanti per completare il board a 5 carte.
    """
    evaluator = Evaluator()
    wins = 0
    ties = 0
    losses = 0
    total_board = len(board_cards)
    # Prepara le carte in formato stringa per treys (ad es. "AH", "2C", ecc.)
    player_str = " ".join(player_cards)
    board_str = " ".join(board_cards) if board_cards else ""
    
    for _ in range(simulations):
        deck = Deck()
        # Rimuove le hole cards e le board cards note dal mazzo
        for card in player_cards:
            deck.cards.remove(Card.new(card))
        for card in board_cards:
            deck.cards.remove(Card.new(card))
        # Estrae 2 carte per l'avversario
        opp_cards = deck.draw(2)
        # Simula le carte mancanti fino a completare il board a 5 carte
        remaining = 5 - total_board
        simulated_board = deck.draw(remaining) if remaining > 0 else []
        
        # Concatena le board note con quelle simulate
        known_board = Card.string_to_cards(board_str) if board_str != "" else []
        final_board = known_board + simulated_board
        
        player_full = Card.string_to_cards(player_str) + final_board
        opp_full = opp_cards + final_board
        
        player_score = evaluator.evaluate([], player_full)
        opp_score = evaluator.evaluate([], opp_full)
        
        if player_score < opp_score:
            wins += 1
        elif player_score == opp_score:
            ties += 1
        else:
            losses += 1
    win_percentage = wins / simulations * 100
    return win_percentage

@socketio.on('frame')
def handle_frame(data):
    """
    Gestisce il frame inviato dal client:
      - Converte il frame da base64 a immagine
      - Rileva le carte e le separa in hole cards e board cards
      - Se sono rilevate almeno 2 hole cards, esegue la simulazione Monte Carlo
      - Invia l'immagine annotata e i risultati (carte, percentuale e mossa) al client
    """
    image = base64_to_image(data)
    predictions, annotated_image, player_preds, board_preds = detect_cards_from_image(image)
    
    # Estrae le etichette delle hole cards (prendiamo le prime 2 se disponibili)
    if len(player_preds) >= 2:
        player_hand_labels = [pred['label'] for pred in player_preds][:2]
    else:
        player_hand_labels = []
    board_hand_labels = [pred['label'] for pred in board_preds]
    
    if len(player_hand_labels) == 2:
        win_percentage = monte_carlo_simulation(player_hand_labels, board_hand_labels)
        if win_percentage > 60:
            move = "Bet/Raise"
        elif win_percentage > 40:
            move = "Check/Call"
        else:
            move = "Fold"
    else:
        win_percentage = None
        move = None

    annotated_base64 = image_to_base64(annotated_image)
    result_data = {
        'annotated': annotated_base64,
        'player_hand': player_hand_labels,
        'board_cards': board_hand_labels,
        'win_percentage': round(win_percentage, 2) if win_percentage is not None else "N/A",
        'move': move if move is not None else "N/A"
    }
    emit('result', result_data)

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port)
