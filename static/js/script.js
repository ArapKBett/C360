async function submitForm(formId, endpoint, resultId) {
    const form = document.getElementById(formId);
    const resultDiv = document.getElementById(resultId);
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(form);
        
        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            if (data.error) {
                resultDiv.innerHTML = `Error: ${data.error}`;
                return;
            }
            
            if (endpoint === '/url') {
                resultDiv.innerHTML = `URL: ${data.url}<br>
                    Status: ${data.is_malicious ? 'Malicious' : 'Safe'}<br>
                    Confidence: ${(data.confidence * 100).toFixed(2)}%`;
            } else if (endpoint === '/network') {
                let output = 'Scan Results:<br>';
                data.result.forEach(host => {
                    output += `Host: ${host.host}<br>`;
                    host.ports.forEach(port => {
                        output += `Port ${port.port} (${port.service}): ${port.state}<br>`;
                    });
                });
                resultDiv.innerHTML = output;
            } else if (endpoint === '/file') {
                resultDiv.innerHTML = `File: ${data.result.file}<br>
                    Hash: ${data.result.current_hash}<br>
                    Status: ${data.result.is_unchanged ? 'Unchanged' : 'Modified'}`;
            } else if (endpoint === '/password') {
                resultDiv.innerHTML = `Score: ${data.result.score}/4<br>
                    Crack Time: ${data.result.crack_time}<br>
                    Feedback: ${data.result.feedback.join(', ') || 'None'}`;
            } else if (endpoint === '/vulnerability') {
                let output = 'Vulnerabilities:<br>';
                data.result.forEach(vuln => {
                    output += `${vuln.vulnerability}: ${vuln.details}<br>`;
                });
                resultDiv.innerHTML = output;
            }
        } catch (error) {
            resultDiv.innerHTML = `Error: ${error.message}`;
        }
    });
}

// Initialize forms
document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('url-form')) submitForm('url-form', '/url', 'url-result');
    if (document.getElementById('network-form')) submitForm('network-form', '/network', 'network-result');
    if (document.getElementById('file-form')) submitForm('file-form', '/file', 'file-result');
    if (document.getElementById('password-form')) submitForm('password-form', '/password', 'password-result');
    if (document.getElementById('vulnerability-form')) submitForm('vulnerability-form', '/vulnerability', 'vulnerability-result');
});
