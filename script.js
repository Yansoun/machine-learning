// Mobile menu toggle
const menuToggle = document.getElementById('menuToggle');
const navMenu = document.getElementById('navMenu');

menuToggle.addEventListener('click', () => {
    navMenu.classList.toggle('active');
});

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            navMenu.classList.remove('active');
        }
    });
});

// Navbar scroll effect
let lastScroll = 0;
window.addEventListener('scroll', () => {
    const navbar = document.getElementById('navbar');
    const currentScroll = window.pageYOffset;
    
    if (currentScroll > 50) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
    
    lastScroll = currentScroll;
});

// Contact form submission
document.getElementById('contactForm').addEventListener('submit', (e) => {
    e.preventDefault();
    alert('Thank you for your message! I will get back to you soon.');
    e.target.reset();
});

// Load projects from projects.json
async function loadProjects() {
    try {
        const response = await fetch('project.json');
        const projects = await response.json();
        displayProjects(projects);
    } catch (error) {
        console.log('Loading projects from inline data...');
        // Fallback to actual projects
        const projects = [
            {
                title: "Sales Prediction System",
                description: "End-to-end time-series forecasting pipeline predicting store sales using historical data, feature engineering, and advanced regression models. Deployed with interactive Streamlit interface.",
                tags: ["Time Series", "XGBoost", "Streamlit"],
                icon: "fa-chart-line",
                github: "https://github.com/Yansoun/machine-learning/tree/project1",
                readme: "sales-prediction"
            },
            {
                title: "Sentiment Analysis Engine",
                description: "NLP system classifying text reviews into positive or negative sentiment using TF-IDF vectorization and machine learning models with real-time prediction capabilities.",
                tags: ["NLP", "TF-IDF", "Scikit-learn"],
                icon: "fa-comment-dots",
                github: "https://github.com/Yansoun/machine-learning/tree/project2",
                readme: "sentiment-analysis"
            },
            {
                title: "Customer Churn Predictor",
                description: "Predictive analytics solution for telecom customer retention using SMOTE for imbalance handling and explainable ML pipeline focused on business ROI optimization.",
                tags: ["Classification", "SMOTE", "XGBoost"],
                icon: "fa-users",
                github: "https://github.com/Yansoun/machine-learning/tree/project-3",
                readme: "customer-churn"
            },
            {
                title: "Food Freshness Detector",
                description: "Computer vision classifier using CNNs and transfer learning with EfficientNet to identify fresh vs rotten food from custom image dataset with deployed prediction app.",
                tags: ["CNN", "Transfer Learning", "Computer Vision"],
                icon: "fa-camera",
                github: "https://github.com/Yansoun/machine-learning/tree/project4",
                readme: "food-freshness"
            },
            {
                title: "Food Calorie Estimation",
                description: "Deep learning app combining MobileNetV2 transfer learning with nutrition data to predict food categories and estimate calories with modern dark-theme UI and visualization.",
                tags: ["Deep Learning", "MobileNet", "Plotly"],
                icon: "fa-utensils",
                github: "https://github.com/Yansoun/machine-learning/tree/project5",
                readme: "calorie-estimation"
            },
            {
                title: "Credit Risk Analyzer",
                description: "Professional ML workflow predicting loan default risk with comprehensive EDA, feature engineering, and deployed credit approval application for financial decision-making.",
                tags: ["Finance", "Random Forest", "Risk Analysis"],
                icon: "fa-credit-card",
                github: "https://github.com/Yansoun/machine-learning/tree/project6",
                readme: "credit-risk"
            },
            {
                title: "Customer Segmentation",
                description: "Unsupervised learning solution using K-Means clustering and PCA to identify distinct market segments with interpretable behavioral patterns and business insights.",
                tags: ["K-Means", "PCA", "Clustering"],
                icon: "fa-project-diagram",
                github: "https://github.com/Yansoun/machine-learning/tree/project7",
                readme: "customer-segmentation"
            },
            {
                title: "A/B Test Analyzer",
                description: "Statistical analysis framework for evaluating experiment effectiveness using hypothesis testing, t-tests, and confidence intervals to drive data-driven business decisions.",
                tags: ["Statistics", "Hypothesis Testing", "Analysis"],
                icon: "fa-flask",
                github: "https://github.com/Yansoun/machine-learning/tree/A_B_Test_Effectiveness_Analysis",
                readme: "ab-test"
            }
        ];
        displayProjects(projects);
    }
}

// Display projects in the grid
function displayProjects(projects) {
    const container = document.getElementById('projectsContainer');
    container.innerHTML = '';
    
    projects.forEach(project => {
        const projectCard = document.createElement('div');
        projectCard.className = 'project-card';
        
        const icon = project.icon || 'fa-code';
        const github = project.github || '#';
        
        projectCard.innerHTML = `
            <div class="project-image">
                <i class="fas ${icon}"></i>
            </div>
            <div class="project-info">
                <h3>${project.title}</h3>
                <p>${project.description}</p>
                <div class="project-tags">
                    ${project.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                </div>
                <div class="project-links">
                    <a href="${github}" target="_blank" class="project-link">
                        <i class="fab fa-github"></i> View Code
                    </a>
                    ${project.readme ? `
                        <a href="#" class="project-link readme-link" data-readme="${project.readme}" data-title="${project.title}">
                            <i class="fas fa-book"></i> Read More
                        </a>
                    ` : ''}
                </div>
            </div>
        `;
        
        container.appendChild(projectCard);
    });
    
    // Add event listeners for README links
    document.querySelectorAll('.readme-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const readmeId = e.currentTarget.getAttribute('data-readme');
            const title = e.currentTarget.getAttribute('data-title');
            openReadmeModal(readmeId, title);
        });
    });
}

// Modal functionality
const modal = document.getElementById('readmeModal');
const closeModal = document.getElementById('closeModal');

closeModal.addEventListener('click', () => {
    modal.classList.remove('active');
});

modal.addEventListener('click', (e) => {
    if (e.target === modal) {
        modal.classList.remove('active');
    }
});

// Open README modal
async function openReadmeModal(readmeId, title) {
    const modalTitle = document.getElementById('modalTitle');
    const modalBody = document.getElementById('modalBody');
    
    modalTitle.textContent = title;
    modalBody.innerHTML = '<p style="text-align: center;">Loading...</p>';
    modal.classList.add('active');
    
    try {
        // Try to fetch from GitHub - adjust the branch/path as needed
        let readmeUrl = '';
        switch(readmeId) {
            case 'sales-prediction':
                readmeUrl = 'https://raw.githubusercontent.com/Yansoun/machine-learning/project1/README.md';
                break;
            case 'sentiment-analysis':
                readmeUrl = 'https://raw.githubusercontent.com/Yansoun/machine-learning/project2/README.md';
                break;
            case 'customer-churn':
                readmeUrl = 'https://raw.githubusercontent.com/Yansoun/machine-learning/project-3/README.md';
                break;
            case 'food-freshness':
                readmeUrl = 'https://raw.githubusercontent.com/Yansoun/machine-learning/project4/README.md';
                break;
            case 'calorie-estimation':
                readmeUrl = 'https://raw.githubusercontent.com/Yansoun/machine-learning/project5/README.md';
                break;
            case 'credit-risk':
                readmeUrl = 'https://raw.githubusercontent.com/Yansoun/machine-learning/project6/README.md';
                break;
            case 'customer-segmentation':
                readmeUrl = 'https://raw.githubusercontent.com/Yansoun/machine-learning/project7/README.md';
                break;
            case 'ab-test':
                readmeUrl = 'https://raw.githubusercontent.com/Yansoun/machine-learning/A_B_Test_Effectiveness_Analysis/README.md';
                break;
        }
        
        const response = await fetch(readmeUrl);
        
        if (response.ok) {
            const text = await response.text();
            modalBody.innerHTML = formatMarkdown(text);
        } else {
            throw new Error('README not found');
        }
    } catch (error) {
        // Fallback content with detailed project descriptions
        modalBody.innerHTML = getSampleReadme(readmeId);
    }
}
// Observe all project cards and skill categories
document.addEventListener('DOMContentLoaded', () => {
    loadProjects();
    
    // Add initial animation states
    const animatedElements = document.querySelectorAll('.project-card, .skill-category, .highlight-item');
    animatedElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
});
// Enhanced README content for actual projects
function getSampleReadme(readmeId) {
    const readmes = {
        'sales-prediction': `
            <h2>Sales Prediction System</h2>
            <p>A comprehensive time-series forecasting solution designed to predict store sales based on historical data and external factors.</p>
            
            <h3>🎯 Project Goals</h3>
            <p>• Build an accurate sales forecasting model<br>
            • Handle seasonality and trends in time-series data<br>
            • Create actionable insights for business planning<br>
            • Deploy an interactive prediction interface</p>
            
            <h3>🛠️ Techniques & Technologies</h3>
            <p><strong>Feature Engineering:</strong> Lag variables, rolling averages, trend decomposition<br>
            <strong>Models:</strong> Linear Regression, RandomForestRegressor, XGBoost<br>
            <strong>Deployment:</strong> Streamlit web application</p>
            
            <h3>📊 Key Features</h3>
            <p>✓ End-to-end time-series pipeline<br>
            ✓ Advanced feature engineering techniques<br>
            ✓ Model comparison and selection<br>
            ✓ Interactive prediction dashboard<br>
            ✓ Strong predictive performance</p>
            
            <h3>💡 Key Learnings</h3>
            <p>This project demonstrated the importance of domain knowledge in feature engineering for time-series problems. Understanding seasonal patterns and business cycles was crucial for model performance.</p>
        `,
        'sentiment-analysis': `
            <h2>Sentiment Analysis Engine</h2>
            <p>An NLP system that classifies text reviews into positive or negative sentiment with high accuracy.</p>
            
            <h3>📝 Overview</h3>
            <p>This project implements a complete text classification pipeline from raw text to deployment, focusing on sentiment analysis of customer reviews.</p>
            
            <h3>🔧 Technical Stack</h3>
            <p><strong>Preprocessing:</strong> Text cleaning, tokenization, stopword removal<br>
            <strong>Vectorization:</strong> TF-IDF (Term Frequency-Inverse Document Frequency)<br>
            <strong>Models:</strong> Logistic Regression, Naive Bayes, Support Vector Machines<br>
            <strong>Deployment:</strong> Real-time Streamlit application</p>
            
            <h3>✨ Highlights</h3>
            <p>✓ Clean and robust preprocessing pipeline<br>
            ✓ High accuracy with linear models<br>
            ✓ Real-time sentiment prediction<br>
            ✓ Interpretable model results</p>
            
            <h3>📈 Performance</h3>
            <p>Achieved excellent classification accuracy through careful text preprocessing and optimal model selection. The system can process and classify sentiment in real-time.</p>
        `,
        'customer-churn': `
            <h2>Customer Churn Predictor</h2>
            <p>A predictive analytics solution helping telecom companies identify customers at risk of leaving.</p>
            
            <h3>🎯 Business Problem</h3>
            <p>Customer churn is expensive. This system predicts which customers are likely to leave, enabling proactive retention strategies.</p>
            
            <h3>🔍 Approach</h3>
            <p><strong>Data Handling:</strong> SMOTE for class imbalance, feature scaling, one-hot encoding<br>
            <strong>Models:</strong> Logistic Regression, Random Forest, XGBoost (best performer)<br>
            <strong>Focus:</strong> Optimizing recall to minimize false negatives<br>
            <strong>Explainability:</strong> Feature importance analysis for business insights</p>
            
            <h3>💼 Business Value</h3>
            <p>✓ Focused on recall to catch at-risk customers<br>
            ✓ Explainable predictions for actionable insights<br>
            ✓ ROI-optimized model selection<br>
            ✓ Deployed web app for instant predictions</p>
            
            <h3>🎓 Key Insights</h3>
            <p>Understanding the business context was crucial. We prioritized recall over accuracy because the cost of losing a customer far exceeds the cost of a retention offer.</p>
        `,
        'food-freshness': `
            <h2>Food Freshness Detection</h2>
            <p>A computer vision solution that classifies food as fresh or rotten using deep learning.</p>
            
            <h3>🖼️ Project Overview</h3>
            <p>This image classification project uses CNNs and transfer learning to automatically detect food freshness, helping reduce food waste.</p>
            
            <h3>🧠 Deep Learning Approach</h3>
            <p><strong>Architecture:</strong> Convolutional Neural Networks (CNNs)<br>
            <strong>Transfer Learning:</strong> EfficientNet / MobileNet pre-trained models<br>
            <strong>Dataset:</strong> Custom collected and cleaned image dataset<br>
            <strong>Training:</strong> Data augmentation, fine-tuning, validation strategies</p>
            
            <h3>🌟 Highlights</h3>
            <p>✓ Built CNN classifier from scratch<br>
            ✓ Applied transfer learning for better performance<br>
            ✓ Created custom food image dataset<br>
            ✓ Deployed simple prediction interface<br>
            ✓ Real-world application addressing food waste</p>
            
            <h3>🚀 Impact</h3>
            <p>This project demonstrates practical computer vision skills and addresses a real-world problem of food waste management.</p>
        `,
        'calorie-estimation': `
            <h2>Food Calorie Estimation App</h2>
            <p>An advanced deep learning application combining food recognition with nutritional analysis.</p>
            
            <h3>🍱 Application Features</h3>
            <p>Upload food images to get instant calorie estimates and complete nutritional breakdowns with confidence scores.</p>
            
            <h3>⚙️ Technical Implementation</h3>
            <p><strong>Model:</strong> MobileNetV2 with transfer learning and fine-tuning<br>
            <strong>Architecture:</strong> CNN for image classification<br>
            <strong>Data:</strong> Integrated nutrition database<br>
            <strong>Visualization:</strong> Plotly charts for macro breakdown<br>
            <strong>UI/UX:</strong> Modern dark-theme Streamlit interface</p>
            
            <h3>✨ Advanced Features</h3>
            <p>✓ Food category prediction from images<br>
            ✓ Calorie and macronutrient estimation<br>
            ✓ Confidence scores for predictions<br>
            ✓ Interactive nutrition charts<br>
            ✓ Professional UI with modern design<br>
            ✓ Full nutrition facts display</p>
            
            <h3>🎨 Design Excellence</h3>
            <p>This project showcases advanced frontend development skills alongside deep learning, creating a polished user experience.</p>
        `,
        'credit-risk': `
            <h2>Credit Risk Analyzer</h2>
            <p>A professional machine learning system for predicting loan default risk in the finance domain.</p>
            
            <h3>💰 Business Context</h3>
            <p>Financial institutions need to assess credit risk accurately. This system predicts loan default probability to support lending decisions.</p>
            
            <h3>📊 Complete ML Workflow</h3>
            <p><strong>EDA:</strong> Comprehensive exploratory data analysis<br>
            <strong>Preprocessing:</strong> Feature scaling, encoding, missing value handling<br>
            <strong>Modeling:</strong> Logistic Regression, Random Forest, XGBoost<br>
            <strong>Evaluation:</strong> ROC curves, confusion matrices, feature importance<br>
            <strong>Deployment:</strong> Credit approval web application</p>
            
            <h3>🎯 Key Results</h3>
            <p>✓ Professional end-to-end pipeline<br>
            ✓ Comprehensive model evaluation<br>
            ✓ Interpretable feature importance<br>
            ✓ User-friendly approval interface<br>
            ✓ High business value demonstration</p>
            
            <h3>💼 Portfolio Strength</h3>
            <p>This project demonstrates ability to handle real-world financial data and build production-ready ML systems with clear business value.</p>
        `,
        'customer-segmentation': `
            <h2>Customer Segmentation Analysis</h2>
            <p>An unsupervised learning project identifying distinct customer groups for targeted marketing strategies.</p>
            
            <h3>🎯 Objective</h3>
            <p>Discover natural customer segments based on behavior patterns to enable personalized marketing and improve business strategy.</p>
            
            <h3>🔬 Methodology</h3>
            <p><strong>Algorithm:</strong> K-Means clustering<br>
            <strong>Dimensionality Reduction:</strong> PCA (Principal Component Analysis)<br>
            <strong>Visualization:</strong> 2D/3D cluster plots<br>
            <strong>Interpretation:</strong> Behavioral pattern analysis</p>
            
            <h3>📈 Deliverables</h3>
            <p>✓ Identified distinct market segments<br>
            ✓ Interpretable customer clusters<br>
            ✓ Clear behavioral patterns<br>
            ✓ Actionable business recommendations<br>
            ✓ Visual segment analysis</p>
            
            <h3>💡 Business Insights</h3>
            <p>The segmentation revealed actionable customer groups with distinct characteristics, enabling targeted marketing strategies and improved customer understanding.</p>
        `,
        'ab-test': `
            <h2>A/B Test Effectiveness Analyzer</h2>
            <p>A statistical analysis framework for evaluating experimental results and making data-driven decisions.</p>
            
            <h3>🧪 Purpose</h3>
            <p>Properly evaluate A/B test results using rigorous statistical methods to determine which version performs better and whether results are significant.</p>
            
            <h3>📊 Statistical Methods</h3>
            <p><strong>Hypothesis Testing:</strong> Null and alternative hypothesis formulation<br>
            <strong>T-Tests:</strong> Two-sample t-tests for mean comparison<br>
            <strong>Statistical Significance:</strong> P-value interpretation<br>
            <strong>Confidence Intervals:</strong> Effect size estimation<br>
            <strong>Sample Size:</strong> Power analysis</p>
            
            <h3>🎓 Key Concepts</h3>
            <p>✓ Understanding Type I and Type II errors<br>
            ✓ Statistical significance vs practical significance<br>
            ✓ Proper experimental design<br>
            ✓ Clear result interpretation<br>
            ✓ Business recommendation formulation</p>
            
            <h3>💼 Career Value</h3>
            <p>This project demonstrates understanding of statistics behind business decisions - a critical skill for data analyst and ML engineer roles. Shows ability to interpret experiments and guide business strategy.</p>
        `
    };
    
    return readmes[readmeId] || '<p>README content not available for this project yet. Please check the GitHub repository for more details.</p>';
}
// Typing effect for hero
const typingText = document.querySelector('.typing-effect');
if (typingText) {
    const text = typingText.textContent;
    typingText.textContent = '';
    let i = 0;
    
    function typeWriter() {
        if (i < text.length) {
            typingText.textContent += text.charAt(i);
            i++;
            setTimeout(typeWriter, 50);
        }
    }
    
    setTimeout(typeWriter, 500);
}
const quizQuestions = [
    {
        question: "Qu'est-ce que JavaScript ?",
        options: [
            "Un langage de programmation côté serveur",
            "Un langage de script côté client",
            "Un framework CSS",
            "Un système de base de données"
        ],
        correct: 1
    },
    {
        question: "Quelle balise HTML est utilisée pour inclure du JavaScript ?",
        options: [
            "<javascript>",
            "<js>",
            "<script>",
            "<code>"
        ],
        correct: 2
    },
    {
        question: "Comment déclare-t-on une variable en JavaScript ?",
        options: [
            "variable x;",
            "var x;",
            "dim x;",
            "int x;"
        ],
        correct: 1
    },
    {
        question: "Quelle méthode affiche un message dans une boîte de dialogue ?",
        options: [
            "alert()",
            "prompt()",
            "console.log()",
            "document.write()"
        ],
        correct: 0
    },
    {
        question: "Comment accède-t-on à un élément avec l'ID 'demo' ?",
        options: [
            "document.getElement('demo')",
            "document.getElementById('demo')",
            "document.id('demo')",
            "getElementById('demo')"
        ],
        correct: 1
    },
    {
        question: "Quel est le résultat de : typeof [1,2,3] ?",
        options: [
            "array",
            "object",
            "list",
            "number"
        ],
        correct: 1
    },
    {
        question: "Comment écrit-on un commentaire sur une ligne en JavaScript ?",
        options: [
            "<!-- commentaire -->",
            "/* commentaire */",
            "// commentaire",
            "# commentaire"
        ],
        correct: 2
    },
    {
        question: "Quelle méthode permet de convertir une chaîne en nombre entier ?",
        options: [
            "Number()",
            "parseInt()",
            "toInteger()",
            "convert()"
        ],
        correct: 1
    },
    {
        question: "Quel événement se déclenche quand on clique sur un bouton ?",
        options: [
            "onPress",
            "onTouch",
            "onClick",
            "onSelect"
        ],
        correct: 2
    },
    {
        question: "Comment déclare-t-on une fonction en JavaScript ?",
        options: [
            "func maFonction()",
            "function: maFonction()",
            "function maFonction()",
            "def maFonction()"
        ],
        correct: 2
    }
];

// Quiz State
let currentQuestionIndex = 0;
let score = 0;
let selectedAnswer = null;

// Get DOM Elements
const quizButton = document.getElementById('quizButton');
const quizModal = document.getElementById('quizModal');
const closeQuiz = document.getElementById('closeQuiz');
const questionSection = document.getElementById('questionSection');
const resultsSection = document.getElementById('resultsSection');
const questionText = document.getElementById('questionText');
const optionsContainer = document.getElementById('optionsContainer');
const nextBtn = document.getElementById('nextBtn');
const progressFill = document.getElementById('progressFill');
const quizProgress = document.getElementById('quizProgress');
const resultEmoji = document.getElementById('resultEmoji');
const resultMessage = document.getElementById('resultMessage');
const scoreNumber = document.getElementById('scoreNumber');
const scorePercentage = document.getElementById('scorePercentage');
const restartBtn = document.getElementById('restartBtn');
const closeResultBtn = document.getElementById('closeResultBtn');

// Open Quiz Modal
quizButton.addEventListener('click', () => {
    quizModal.classList.add('active');
    resetQuiz();
    loadQuestion();
});

// Close Quiz Modal
closeQuiz.addEventListener('click', () => {
    quizModal.classList.remove('active');
});

// Close modal when clicking outside
quizModal.addEventListener('click', (e) => {
    if (e.target === quizModal) {
        quizModal.classList.remove('active');
    }
});

// Load Question
function loadQuestion() {
    const question = quizQuestions[currentQuestionIndex];
    
    // Update progress
    quizProgress.textContent = `Question ${currentQuestionIndex + 1} sur ${quizQuestions.length}`;
    progressFill.style.width = `${((currentQuestionIndex + 1) / quizQuestions.length) * 100}%`;
    
    // Display question
    questionText.textContent = question.question;
    
    // Clear previous options
    optionsContainer.innerHTML = '';
    
    // Create option buttons
    question.options.forEach((option, index) => {
        const optionBtn = document.createElement('button');
        optionBtn.className = 'option-btn';
        optionBtn.textContent = option;
        optionBtn.addEventListener('click', () => selectAnswer(index));
        optionsContainer.appendChild(optionBtn);
    });
    
    // Reset next button
    nextBtn.disabled = true;
    selectedAnswer = null;
}

// Select Answer
function selectAnswer(answerIndex) {
    selectedAnswer = answerIndex;
    
    // Remove selected class from all options
    const allOptions = document.querySelectorAll('.option-btn');
    allOptions.forEach(btn => btn.classList.remove('selected'));
    
    // Add selected class to clicked option
    allOptions[answerIndex].classList.add('selected');
    
    // Enable next button
    nextBtn.disabled = false;
}

// Next Question
nextBtn.addEventListener('click', () => {
    // Check if answer is correct
    if (selectedAnswer === quizQuestions[currentQuestionIndex].correct) {
        score++;
    }
    
    // Move to next question or show results
    currentQuestionIndex++;
    
    if (currentQuestionIndex < quizQuestions.length) {
        loadQuestion();
    } else {
        showResults();
    }
});

// Show Results
function showResults() {
    questionSection.style.display = 'none';
    resultsSection.style.display = 'block';
    
    // Calculate percentage
    const percentage = (score / quizQuestions.length) * 100;
    
    // Display score
    scoreNumber.textContent = `${score} / ${quizQuestions.length}`;
    scorePercentage.textContent = `Score: ${Math.round(percentage)}%`;
    
    // Set emoji and message based on score
    if (percentage === 100) {
        resultEmoji.textContent = '🏆';
        resultMessage.textContent = '🎉 Parfait ! Vous êtes un expert JavaScript !';
    } else if (percentage >= 80) {
        resultEmoji.textContent = '🌟';
        resultMessage.textContent = '🌟 Excellent ! Très bonne maîtrise !';
    } else if (percentage >= 60) {
        resultEmoji.textContent = '👍';
        resultMessage.textContent = '👍 Bien ! Vous avez de bonnes bases !';
    } else if (percentage >= 40) {
        resultEmoji.textContent = '📚';
        resultMessage.textContent = '📚 Pas mal ! Continuez à apprendre !';
    } else {
        resultEmoji.textContent = '💪';
        resultMessage.textContent = '💪 Courage ! Révisez et réessayez !';
    }
}

// Restart Quiz
restartBtn.addEventListener('click', () => {
    resultsSection.style.display = 'none';
    questionSection.style.display = 'block';
    resetQuiz();
    loadQuestion();
});

// Close Results
closeResultBtn.addEventListener('click', () => {
    quizModal.classList.remove('active');
});

// Reset Quiz
function resetQuiz() {
    currentQuestionIndex = 0;
    score = 0;
    selectedAnswer = null;
    questionSection.style.display = 'block';
    resultsSection.style.display = 'none';
}
