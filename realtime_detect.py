import pyshark
import pandas as pd
import pickle

# Load trained model + scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

print("🔴 Starting real-time MITM detection...\n")

# Capture live packets from interface
capture = pyshark.LiveCapture(interface='eth0')

for packet in capture.sniff_continuously(packet_count=50):

    try:
        # Extract features
        packet_size = int(packet.length)

        if hasattr(packet, 'tcp'):
            src_port = int(packet.tcp.srcport)
            dst_port = int(packet.tcp.dstport)
            protocol = 6  # TCP
        else:
            continue  # skip non-TCP packets

        # Create dataframe
        data = pd.DataFrame(
            [[packet_size, src_port, dst_port, protocol]],
            columns=['packet_size','src_port','dst_port','protocol']
        )

        # Scale input
        data_scaled = scaler.transform(data)

        # Predict
        prediction = model.predict(data_scaled)

        # Output
        if prediction[0] == 1:
            print(f"🚨 ATTACK DETECTED | Size:{packet_size} | {src_port}->{dst_port}")
        else:
            print(f"✅ SAFE | Size:{packet_size} | {src_port}->{dst_port}")

    except:
        continue
