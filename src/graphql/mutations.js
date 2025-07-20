import { gql } from '@apollo/client';

export const PREDICT_SENTIMENT = gql`
  mutation PredictSentiment($text: String!) {
    predictSentiment(text: $text) {
      label
      score
    }
  }
`;
