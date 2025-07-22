import React from 'react';
import { ThemeProvider } from './contexts/ThemeContext';
import { Header } from './components/Header';
import { SentimentAnalyzer } from './components/SentimentAnalyzer';
import { Footer } from './components/Footer';

function App() {
    return (
        <ThemeProvider>
            <div className="min-h-screen transition-colors duration-300
                    bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50
                    dark:from-black dark:via-gray-900 dark:to-gray-800">

                <div className="fixed inset-0 overflow-hidden pointer-events-none">
                    <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-blue-400/20 to-purple-600/20 rounded-full blur-3xl animate-pulse"></div>
                    <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-br from-purple-400/20 to-pink-600/20 rounded-full blur-3xl animate-pulse delay-1000"></div>
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-gradient-to-br from-blue-300/10 to-purple-400/10 rounded-full blur-3xl animate-pulse delay-2000"></div>
                </div>

                <div className="fixed inset-0 bg-white/30 dark:bg-black/30 backdrop-blur-sm pointer-events-none"></div>

                <div className="relative z-10 flex flex-col min-h-screen">
                    <Header />

                    <main className="flex-1 container mx-auto px-6 py-12">
                        <div className="max-w-4xl mx-auto text-center mb-12">
                            <h2 className="text-4xl md:text-5xl font-bold text-indigo-700 dark:text-indigo-300 mb-4">
                                Real-Time Sentiment Analysis

                            </h2>
                            <p className="text-lg text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
                                Powered by a fine-tuned Twitter RoBERTa model for advanced emotion detection.
                                Instant insights with confidence scores.
                            </p>
                        </div>

                        <div className="relative">
                            <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-purple-600/20 rounded-2xl blur-xl"></div>

                            <div className="relative bg-white/20 dark:bg-gray-900/40 backdrop-blur-sm 
                            rounded-2xl border border-white/30 dark:border-gray-600/50 
                            shadow-xl p-8">
                                <SentimentAnalyzer />
                            </div>
                        </div>

                        <div className="mt-16 grid md:grid-cols-3 gap-8">
                            <div className="text-center p-6 rounded-xl 
                            bg-white/60 dark:bg-gray-800/40 backdrop-blur-sm
                            border border-gray-200/60 dark:border-gray-700/50
                            shadow-lg hover:shadow-xl transition-all duration-300">
                                <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg mx-auto mb-4 flex items-center justify-center">
                                    <span className="text-white font-bold text-sm">🤖</span>
                                </div>
                                <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-2">
                                    RoBERTa Base Model
                                </h3>
                                <p className="text-sm text-gray-600 dark:text-gray-400">
                                    Using cardiffnlp/twitter-roberta-base-sentiment-latest pre-trained model
                                </p>
                            </div>

                            <div className="text-center p-6 rounded-xl 
                            bg-white/60 dark:bg-gray-800/40 backdrop-blur-sm
                            border border-gray-200/60 dark:border-gray-700/50
                            shadow-lg hover:shadow-xl transition-all duration-300">
                                <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-600 rounded-lg mx-auto mb-4 flex items-center justify-center">
                                    <span className="text-white font-bold text-sm">⚡</span>
                                </div>
                                <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-2">
                                    LoRA Fine-Tuning
                                </h3>
                                <p className="text-sm text-gray-600 dark:text-gray-400">
                                    Parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA)
                                </p>
                            </div>

                            <div className="text-center p-6 rounded-xl 
                            bg-white/60 dark:bg-gray-800/40 backdrop-blur-sm
                            border border-gray-200/60 dark:border-gray-700/50
                            shadow-lg hover:shadow-xl transition-all duration-300">
                                <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-blue-600 rounded-lg mx-auto mb-4 flex items-center justify-center">
                                    <span className="text-white font-bold text-sm">🔧</span>
                                </div>
                                <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-2">
                                    4-bit Quantization
                                </h3>
                                <p className="text-sm text-gray-600 dark:text-gray-400">
                                    Memory-efficient training with 4-bit quantization using bitsandbytes
                                </p>
                            </div>
                        </div>
                    </main>

                    <Footer />
                </div>
            </div>
        </ThemeProvider>
    );
}

export default App;
