import struct

PACKET_HEADER_SIZE = 8
PACKET_BODY_SIZE = 64
PACKET_SIZE = PACKET_HEADER_SIZE + PACKET_BODY_SIZE

PRODUCT_ID_CUGOV4 = 1
PRODUCT_ID_CUGOV3I = 1
PRODUCT_ID_UNAJU = 10

ROBOT_ID = 8888


def calculate_checksum(data: bytes) -> int:
    s = 0
    for i in range(0, len(data), 2):
        word = struct.unpack('<H', data[i:i+2])[0]
        s += word
        while s >> 16:
            s = (s & 0xFFFF) + (s >> 16)
    return (~s) & 0xFFFF


def build_send_packet(l_rpm: float, r_rpm: float, product_id: int = PRODUCT_ID_CUGOV4) -> bytes:
    body = bytearray(PACKET_BODY_SIZE)
    struct.pack_into('<ff', body, 0, l_rpm, r_rpm)
    checksum = calculate_checksum(bytes(body))
    header = struct.pack('<HHHH', product_id, ROBOT_ID, PACKET_SIZE, checksum)
    return bytes(header + body)


def parse_receive_packet(packet: bytes) -> dict:
    if len(packet) != PACKET_SIZE:
        raise ValueError(f'expected {PACKET_SIZE} bytes, got {len(packet)}')
    body = packet[PACKET_HEADER_SIZE:]
    header = packet[:PACKET_HEADER_SIZE]
    product_id, robot_id, length, checksum = struct.unpack('<HHHH', header)
    calc_cs = calculate_checksum(body)
    if calc_cs != checksum:
        raise ValueError(f'checksum mismatch: recv={checksum:04x} calc={calc_cs:04x}')
    encoder_l, encoder_r = struct.unpack_from('<ii', body, 0)
    run_mode = body[8]
    return {'product_id': product_id, 'robot_id': robot_id,
            'encoder_l': encoder_l, 'encoder_r': encoder_r, 'run_mode': run_mode}


def cobs_encode(data: bytes) -> bytes:
    out = bytearray()
    out.append(0)
    code_idx = 0
    code = 1

    for byte in data:
        if byte == 0:
            out[code_idx] = code
            code_idx = len(out)
            out.append(0)
            code = 1
        else:
            out.append(byte)
            code += 1
            if code == 0xFF:
                out[code_idx] = code
                code_idx = len(out)
                out.append(0)
                code = 1

    out[code_idx] = code
    out.append(0)
    return bytes(out)


def cobs_decode(data: bytes) -> bytes:
    out = bytearray()
    i = 0
    while i < len(data):
        code = data[i]
        if code == 0:
            break
        i += 1
        out.extend(data[i:i + code - 1])
        i += code - 1
        if code != 0xFF and i < len(data) and data[i] != 0:
            out.append(0)
    return bytes(out)
