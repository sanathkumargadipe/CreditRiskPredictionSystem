def get_recommendation(prob):

    if prob > 0.7:
        return "Reject Loan or Approve with Reduced Amount"

    elif prob > 0.4:
        return "Approve with Caution"

    else:
        return "Loan Can Be Approved Safely"