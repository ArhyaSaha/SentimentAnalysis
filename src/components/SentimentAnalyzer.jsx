import React, { useState, useEffect, useCallback } from 'react';
import { useMutation } from '@apollo/client';
import { Send, TrendingUp, TrendingDown, Loader2, Zap } from 'lucide-react';
import { PREDICT_SENTIMENT } from '../graphql/mutations';

export const SentimentAnalyzer = () => {
    const [text, setText] = useState('');
    const [result, setResult] = useState(null);
    const [isRealTimeEnabled, setIsRealTimeEnabled] = useState(true);

    const [predictSentiment, { loading, error }] = useMutation(PREDICT_SENTIMENT, {
        onCompleted: (data) => {
            setResult(data.predictSentiment);
        },
    });

    const debouncedAnalysis = useCallback(
        debounce(async (textToAnalyze) => {
            if (textToAnalyze.trim() && textToAnalyze.length > 10) {
                try {
                    await predictSentiment({ variables: { text: textToAnalyze } });
                } catch (err) {
                    console.error('Real-time prediction error:', err);
                }
            }
        }, 1000),
        [predictSentiment]
    );

    useEffect(() => {
        if (isRealTimeEnabled && text.trim()) {
            debouncedAnalysis(text);
        }
        return () => {
            debouncedAnalysis.cancel && debouncedAnalysis.cancel();
        };
    }, [text, isRealTimeEnabled, debouncedAnalysis]);

    function debounce(func, wait) {
        let timeout;
        const debounced = function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
        debounced.cancel = () => clearTimeout(timeout);
        return debounced;
    }

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!text.trim()) return;

        try {
            await predictSentiment({ variables: { text } });
        } catch (err) {
            console.error('Prediction error:', err);
        }
    };

    const getSentimentIcon = (label) => {
        return label === 'positive' ?
            <TrendingUp className="w-6 h-6 text-green-500" /> :
            <TrendingDown className="w-6 h-6 text-red-500" />;
    };

    const getSentimentColor = (label) => {
        return label === 'positive' ? 'text-green-500' : 'text-red-500';
    };

    const CircularMeter = ({ sentiment, score }) => {
        const radius = 120;
        const strokeWidth = 10;
        const normalizedRadius = radius - strokeWidth * 2;
        const circumference = normalizedRadius * 2 * Math.PI;
        const strokeDasharray = `${circumference} ${circumference}`;
        const strokeDashoffset = circumference - (score * circumference);

        const getSentimentGradient = (label) => {
            return label === 'positive' ?
                'from-green-400 to-emerald-500' :
                'from-red-400 to-rose-500';
        };

        const getSentimentColor = (label) => {
            return label === 'positive' ? '#10b981' : '#ef4444';
        };

        return (
            <div className="flex flex-col items-center">
                <div className="relative w-64 h-64">
                    <svg
                        height={radius * 2}
                        width={radius * 2}
                        className="transform -rotate-90"
                    >
                        <circle
                            stroke="#e5e7eb"
                            fill="transparent"
                            strokeWidth={strokeWidth}
                            r={normalizedRadius}
                            cx={radius}
                            cy={radius}
                            className="dark:stroke-gray-600"
                        />
                        <circle
                            stroke={getSentimentColor(sentiment)}
                            fill="transparent"
                            strokeWidth={strokeWidth}
                            strokeDasharray={strokeDasharray}
                            strokeDashoffset={strokeDashoffset}
                            strokeLinecap="round"
                            r={normalizedRadius}
                            cx={radius}
                            cy={radius}
                            className="transition-all duration-1000 ease-out"
                            style={{
                                filter: 'drop-shadow(0 0 8px rgba(59, 130, 246, 0.5))',
                            }}
                        />
                    </svg>

                    <div className="absolute inset-0 flex flex-col items-center justify-center">
                        <div className="text-center pl-3 pr-6 w-full mb-4">
                            <div className='flex items-center justify-center gap-x-1'>
                                <div className='mt-1'>
                                    {getSentimentIcon(sentiment)}
                                </div>
                                <p className={`text-2xl font-bold capitalize mt-2 bg-gradient-to-r ${getSentimentGradient(sentiment)} bg-clip-text text-transparent`}>
                                    {sentiment}
                                </p>
                            </div>
                            <p className="text-4xl font-bold text-gray-800 dark:text-white mt-1 ml-2">
                                {(score * 100).toFixed(0)}%
                            </p>
                            <p className="text-sm text-gray-500 dark:text-gray-400 pt-1">
                                Confidence
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        );
    };

    return (
        <div className="w-full max-w-2xl mx-auto">
            <div className="mb-4 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Zap className="w-5 h-5 text-blue-500" />
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        Real-time Analysis
                    </span>
                </div>
                <button
                    onClick={() => setIsRealTimeEnabled(!isRealTimeEnabled)}
                    className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${isRealTimeEnabled ? 'bg-blue-600' : 'bg-gray-200 dark:bg-gray-700'
                        }`}
                >
                    <span
                        className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${isRealTimeEnabled ? 'translate-x-5' : 'translate-x-0'
                            }`}
                    />
                </button>
            </div>

            {/* Helpful tip
            {isRealTimeEnabled && (
                <div className="mb-4 p-3 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-700 dark:text-blue-300">
                    <p className="text-sm">
                        💡 Real-time mode is active. Analysis will start automatically after typing 10+ characters.
                    </p>
                </div>
            )} */}

            {result && (
                <div className="mb-6 p-6 rounded-xl 
                      bg-white/10 dark:bg-gray-800/20 backdrop-blur-sm
                      border border-white/30 dark:border-gray-700/50
                      shadow-lg">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-100">
                            Sentiment Analysis
                        </h3>
                        {isRealTimeEnabled && (
                            <div className="flex items-center gap-2 text-sm text-blue-500">
                                <Zap className="w-4 h-4" />
                                <span>Real-time</span>
                            </div>
                        )}
                    </div>

                    <CircularMeter sentiment={result.label} score={result.score} />
                </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-6">
                <div className="relative">
                    <textarea
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        placeholder={isRealTimeEnabled ?
                            "Start typing to see real-time sentiment analysis..." :
                            "Enter your text here to analyze sentiment..."
                        }
                        className="w-full h-32 p-4 rounded-xl resize-none 
                     bg-white/10 dark:bg-gray-800/40 backdrop-blur-sm
                     border border-white/30 dark:border-gray-600/50
                     text-gray-800 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400
                     focus:outline-none focus:ring-2 focus:ring-blue-500/50
                     transition-all duration-300"
                        disabled={loading}
                    />
                    <div className="absolute bottom-4 right-4 flex items-center gap-2">
                        {isRealTimeEnabled && text.length > 10 && loading && (
                            <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
                        )}
                        <span className="text-sm text-gray-500">
                            {text.length}/1000
                        </span>
                    </div>
                </div>

                <button
                    type="submit"
                    disabled={!text.trim() || loading || (isRealTimeEnabled && text.length > 10)}
                    className={`w-full py-3 px-6 rounded-xl font-medium text-white shadow-lg hover:shadow-xl
                   transform transition-all duration-300 hover:scale-105
                   disabled:opacity-50 disabled:cursor-not-allowed
                   disabled:transform-none
                   flex items-center justify-center gap-2 ${isRealTimeEnabled ?
                            'bg-gradient-to-r from-gray-500 to-gray-600' :
                            'bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700'
                        }`}
                >
                    {loading && !isRealTimeEnabled ? (
                        <>
                            <Loader2 className="w-5 h-5 animate-spin" />
                            Analyzing...
                        </>
                    ) : isRealTimeEnabled ? (
                        <>
                            <Zap className="w-5 h-5" />
                            Real-time Mode Active
                        </>
                    ) : (
                        <>
                            <Send className="w-5 h-5" />
                            Analyze Sentiment
                        </>
                    )}
                </button>
            </form>

            {error && (
                <div className="mt-6 p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-600 dark:text-red-400">
                    <p className="font-medium">Error analyzing sentiment:</p>
                    <p className="text-sm mt-1">{error.message}</p>
                </div>
            )}
        </div>
    );
};
