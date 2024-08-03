document.getElementById('query-form').addEventListener('submit', function(e) {
    e.preventDefault();
    const query = document.getElementById('query').value;

    // Show loading indicator
    document.getElementById('loading').style.display = 'block';
    
    fetch('/query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({ 'query': query })
    })
    .then(response => response.json())
    .then(data => {
        // Hide loading indicator
        document.getElementById('loading').style.display = 'none';
    
        document.getElementById('answer').innerHTML = data.answer.replace(/\n/g, '<br>').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');;
        document.getElementById('rag-query').innerHTML = data.rag_query.replace(/\n/g, '<br>');

        const topParagraphsDiv = document.getElementById('top-paragraphs');
        topParagraphsDiv.innerHTML = '';
        data.top_paragraphs.forEach((item, index) => {
            const p = document.createElement('p');
            p.innerHTML = `<strong>Match #${index + 1}</strong><br>${item.replace(/\n/g, '<br>')}`;
            topParagraphsDiv.appendChild(p);
        });
    });
});
