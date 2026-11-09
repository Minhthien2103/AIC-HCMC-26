//------------------------ Random search ------------------------//

function generateRandomString(length = 10) {
    const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    for (let i = 0; i < length; i++) {
        result += characters.charAt(Math.floor(Math.random() * characters.length));
    }
    return result;
}


// Handle dice click function
async function handleDiceClick() {
    const diceIcon = document.getElementById("dice-icon");
    diceIcon.parentElement.classList.add('shake');
    
    setTimeout(() => {
        diceIcon.parentElement.classList.remove('shake');
    }, 200); // Remove the class after the animation duration

    const randomQuery = generateRandomString();
    document.getElementById("Text-Query-First").value = randomQuery;
    document.getElementById("Text-Query-Second").value = "";
    toggleLoadingIndicator(true)

    if (isQuickSearch) {
        // Quick search mode
        await performPagnitionCombinedSearch();
    } else {
        // Normal search mode
        await performCombinedSearch();
    }
    toggleLoadingIndicator(false)
}


function handleKeyboardShortcuts(event) {
    if (event.ctrlKey && event.key === 'd') {
        event.preventDefault(); // Prevent the default browser action
        handleDiceClick();
    }
}

// Add event listeners when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    const diceIcon = document.querySelector(".dice-logo img");
    if (diceIcon) {
        diceIcon.addEventListener("click", handleDiceClick);
    }
    document.addEventListener('keydown', handleKeyboardShortcuts);
});