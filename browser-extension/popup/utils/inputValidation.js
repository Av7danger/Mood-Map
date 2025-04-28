// Utility function to validate user input
export function validateInput(input) {
    if (!input || input.trim() === '') {
        return { isValid: false, message: 'Input cannot be empty.' };
    }
    if (input.length > 500) {
        return { isValid: false, message: 'Input exceeds the maximum length of 500 characters.' };
    }
    return { isValid: true, message: '' };
}