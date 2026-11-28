"""Operation receipts committed in the same JSON document as the write."""
import hashlib
import time


def fingerprint(content):
    return hashlib.sha256(str(content).encode('utf-8')).hexdigest()


def previous(data, operation_id, content, *, namespace='write'):
    if not operation_id:
        return None
    receipt = data.get('_operation_receipts', {}).get(f'{namespace}:{operation_id}')
    if receipt is not None:
        if receipt.get('sha256') != fingerprint(content):
            raise ValueError('operation_id reused for different content')
        return dict(receipt['result'], duplicate=True)
    return None


def record(data, operation_id, content, result, *, namespace='write'):
    if operation_id:
        now = time.time()
        data.setdefault('_operation_receipts', {})[f'{namespace}:{operation_id}'] = {
            'sha256': fingerprint(content), 'result': result, 'at': now,
            'expires_at': now + 14 * 86400}
