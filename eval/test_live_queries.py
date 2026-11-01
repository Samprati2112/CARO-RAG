import urllib.request
import json

def test_q(q):
    req = urllib.request.Request(
        'http://127.0.0.1:8000/api/query',
        data=json.dumps({'question': q}).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        print('='*70)
        print('QUERY:', q[:60])
        print('GOLD:', res.get('gold_answer'))
        c_acc = res['caro']['accuracy']
        c_conf = res['caro']['confidence']
        b_acc = res['baseline']['accuracy']
        b_conf = res['baseline']['confidence']
        print(f"CARO: {res['caro']['answer']} (Acc: {c_acc}%, Conf: {c_conf}%)")
        print(f"BASE: {res['baseline']['answer']} (Acc: {b_acc}%, Conf: {b_conf}%)")
        print('WINNER:', res['comparison']['winner'])
        print('REASON:', res['comparison']['reason'])

test_q('25137. Which company produced both the movie Cinderella and the movie The Wild Country.')
test_q("25136. What Italian pop music duo stared in the 1968 film L'oro del mondo directed by Aldo Grimaldi?")
test_q("25135. What was the profession of the person by who's name a memoir exists and the film Stronger is made?")
test_q('501. What Category 5 hurricane cause the Sqannanoa River to flood?')
test_q('Who was the lead singer of Queen?')
