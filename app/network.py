import nmap

def scan_network(target):
    nm = nmap.PortScanner()
    try:
        nm.scan(target, arguments='-sS -T4 -p-')  # TCP SYN scan, aggressive timing
        result = []
        for host in nm.all_hosts():
            host_info = {'host': host, 'ports': []}
            for proto in nm[host].all_protocols():
                ports = nm[host][proto].keys()
                for port in ports:
                    state = nm[host][proto][port]['state']
                    service = nm[host][proto][port]['name']
                    host_info['ports'].append({'port': port, 'state': state, 'service': service})
            result.append(host_info)
        return result
    except Exception as e:
        raise Exception(f"Network scan failed: {str(e)}")
